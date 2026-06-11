# src/htr_sp5/store.py
"""PostgreSQL persistence for SP-5 (eval runs/results + upload history).

Same shape as htr_sp3.store.PgVectorStore: psycopg v3 is imported lazily so importing this module
never requires a DB driver; every method opens a short-lived connection via _connect().

Three tables managed by this store
------------------------------------
eval_run
    One row per batch evaluation job.  Groups many per-image results together.  Contains metadata
    about what dataset was evaluated, how many samples, which model checkpoint, and whether the
    RAG pipeline was enabled.  The dashboard uses this table to populate the "run selector" drop-down.

eval_result
    One row per (run, image, scenario) triple — i.e. four rows per image for a 4-scenario run.
    Long format (one row per scenario rather than one row with four columns) so ``eval_summary``
    can aggregate across scenarios with a single ``GROUP BY scenario`` query.  Foreign-keyed to
    ``eval_run`` with ``ON DELETE CASCADE`` so deleting a run cleans up all its results atomically.

upload_result
    One row per end-user image upload on the single-image inference page.  Stores the original
    filename, the MinIO object key (so the image can be fetched later for the presigned-URL endpoint),
    an optional ground-truth string, and the M1–M4 inference results as a JSONB blob.

Relationship to SP-3
---------------------
Both SP-3 (vocab/pgvector) and SP-5 write to the *same* Postgres instance, sharing HTR_PG_DSN.
The tables are kept separate (no FK between them) so SP-3 can be deployed and tested independently.

psycopg v3 idiom (mirroring htr_sp3.store.PgVectorStore)
----------------------------------------------------------
    def _connect(self):
        import psycopg              # lazy import — no driver needed at import time
        return psycopg.connect(self._dsn)

    # Usage: always ``with self._connect() as conn, conn.cursor() as cur:``
    # The context manager on the connection rolls back on exception; explicit conn.commit()
    # is needed to persist changes (psycopg v3 auto-begins a transaction on first statement).
"""
from __future__ import annotations

import json
from typing import Iterable

# htr_sp5.config holds PG_DSN and the three table-name constants.  Imported here (not inside
# methods) because reading table names at import time is cheap and safe — no DB connection is
# opened, only the string constants are resolved.
from htr_sp5 import config
from htr_sp5.schemas import EvalResultRow


class Sp5Store:
    """Persistence layer for SP-5 eval runs, per-sample results, and upload history.

    Mirrors the PgVectorStore class from htr_sp3: a thin wrapper around psycopg v3 that opens a
    short-lived connection per method call.  There is intentionally no persistent connection pool
    here — the batch evaluator runs infrequently enough that a new connection per INSERT batch
    adds negligible overhead compared to inference latency.

    Design decision — no ORM:
        Raw SQL via psycopg v3 is used throughout.  An ORM (SQLAlchemy, Tortoise, etc.) would add
        a heavy dependency and obscure the schema, which must be fully legible for the thesis.

    Design decision — short-lived connections:
        Each public method opens a fresh connection with ``_connect()`` and closes it in the
        ``with`` block.  This is safe for the thesis workload (low QPS) and avoids managing a
        connection-pool lifecycle.  The FastAPI server (later tasks) may wrap this store with a
        lifespan-scoped connection if needed.

    Attributes:
        _dsn: The PostgreSQL connection string, e.g. ``"postgresql://user:pass@host:5432/db"``.
    """

    def __init__(self, dsn: str | None = None) -> None:
        """Initialise the store with an optional DSN override.

        If ``dsn`` is ``None``, the DSN is read from ``htr_sp5.config.PG_DSN``, which itself reads
        the ``HTR_PG_DSN`` environment variable (with a localhost fallback for dev).  Passing an
        explicit ``dsn`` is used by tests and integration scripts that need to target a specific DB.

        No connection is attempted here — psycopg is not even imported at this point.  This keeps
        the store cheap to construct in tests and in the FastAPI app startup path.

        Args:
            dsn: PostgreSQL connection string.  ``None`` → use ``config.PG_DSN``.
        """
        # Store the DSN as an instance attribute so tests can assert on it without triggering a
        # real DB connection (see test_store_constructs_from_config_without_connecting).
        self._dsn = dsn or config.PG_DSN

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """Open and return a new psycopg v3 connection to the configured DSN.

        psycopg is imported here (not at module top level) so that the rest of the codebase can
        ``import htr_sp5.store`` without the psycopg package being installed.  This matches the
        lazy-import pattern in htr_sp3.store.PgVectorStore._connect.

        The returned connection is intended to be used as a context manager:

            with self._connect() as conn, conn.cursor() as cur:
                ...
                conn.commit()

        psycopg v3 context managers roll back automatically on exception, so any error raised
        inside the ``with`` block leaves the DB in a consistent state.

        Returns:
            A psycopg ``Connection`` object.
        """
        import psycopg  # lazy import: psycopg is only needed when an actual DB call is made
        return psycopg.connect(self._dsn)

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create the three SP-5 tables and the aggregation index (idempotent; never truncates).

        Safe to call multiple times: ``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT
        EXISTS`` are used throughout.  Unlike PgVectorStore.create_schema(), this method does NOT
        truncate existing data — batch eval runs and upload history should accumulate across runs.

        Tables created:
            ``config.EVAL_RUN_TABLE``    — one row per batch evaluation job.
            ``config.EVAL_RESULT_TABLE`` — one row per (run, sample, scenario) triple.
            ``config.UPLOAD_TABLE``      — one row per end-user image upload.

        Index created:
            ``{eval_result_table}_run_scenario`` — B-tree index on (run_id, scenario).  Makes
            ``eval_summary``'s ``GROUP BY scenario WHERE run_id = %s`` fast even for large runs.
        """
        with self._connect() as conn, conn.cursor() as cur:
            # ------------------------------------------------------------------
            # eval_run: one row per batch job
            # ------------------------------------------------------------------
            # id           : auto-increment PK; returned to callers so they can associate results.
            # created_at   : auto-set to now(); used to order runs in the dashboard drop-down.
            # dataset      : human-readable dataset name (e.g. "iam-line-test").
            # n_samples    : total samples in the run; stored here so the API doesn't have to COUNT.
            # model_ref    : optional checkpoint identifier (e.g. a RunPod job ID or git SHA).
            # rag_enabled  : whether the RAG corrector was active for this run.
            # notes        : free-text field for thesis annotations.
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.EVAL_RUN_TABLE} (
                    id          BIGSERIAL PRIMARY KEY,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    dataset     TEXT        NOT NULL,
                    n_samples   INT         NOT NULL,
                    model_ref   TEXT,
                    rag_enabled BOOLEAN     NOT NULL,
                    notes       TEXT
                )""")

            # ------------------------------------------------------------------
            # eval_result: one row per (run, sample, scenario) triple
            # ------------------------------------------------------------------
            # run_id       : FK to eval_run with CASCADE DELETE — removes all results when a run
            #                row is deleted (keeps orphan cleanup simple).
            # sample_id    : caller-assigned identifier for the source image (e.g. a UUID or
            #                the original filename).
            # scenario     : "m1" | "m2" | "m3" | "m4" — the four HTR pipeline variants.
            # text         : raw HTR output for this scenario.
            # ground_truth : reference transcription; NULL when the user did not supply one.
            # cer / wer    : error rates as percentages; NULL when ground_truth is NULL.
            # latency_s    : wall-clock seconds from scenario start to first byte of output.
            # log          : short status string from the orchestrator (e.g. "Inference OK").
            # status_tag   : UI badge label (e.g. "Raw Output", "CoT Output", "Corrected").
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.EVAL_RESULT_TABLE} (
                    id               BIGSERIAL PRIMARY KEY,
                    run_id           BIGINT  NOT NULL
                                         REFERENCES {config.EVAL_RUN_TABLE}(id) ON DELETE CASCADE,
                    sample_id        TEXT    NOT NULL,
                    scenario         TEXT    NOT NULL,
                    text             TEXT,
                    ground_truth     TEXT,
                    cer              REAL,
                    wer              REAL,
                    latency_seconds  REAL,
                    log              TEXT,
                    status_tag       TEXT
                )""")

            # B-tree index on (run_id, scenario) so the per-run GROUP BY in eval_summary() is an
            # index scan rather than a full table scan — important when the table holds many runs.
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {config.EVAL_RESULT_TABLE}_run_scenario "
                f"ON {config.EVAL_RESULT_TABLE} (run_id, scenario)")

            # ------------------------------------------------------------------
            # upload_result: one row per end-user image upload
            # ------------------------------------------------------------------
            # id           : auto-increment PK; returned so the API can build a presigned-URL
            #                endpoint URL (/uploads/{id}/image).
            # created_at   : auto-set; used for the chronological history view.
            # filename     : original client filename (e.g. "receipt.png") — display only.
            # object_key   : MinIO object key (e.g. "uploads/uuid.png") — used to generate
            #                presigned GET URLs.  Fetched by get_upload_object_key().
            # ground_truth : optional reference string supplied by the user.
            # results      : JSONB blob — the folded {model: {text, cer, wer, ...}} dict from
            #                htr_sp5.schemas.fold_results().  Stored verbatim; no normalisation
            #                needed for the history view.
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.UPLOAD_TABLE} (
                    id           BIGSERIAL PRIMARY KEY,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    filename     TEXT        NOT NULL,
                    object_key   TEXT        NOT NULL,
                    ground_truth TEXT,
                    results      JSONB       NOT NULL
                )""")

            # Commit all three DDL statements as a single transaction so we never end up with a
            # partial schema (e.g. eval_run created but eval_result missing).
            conn.commit()

    # ------------------------------------------------------------------
    # eval_run CRUD
    # ------------------------------------------------------------------

    def create_eval_run(
        self,
        dataset: str,
        n_samples: int,
        model_ref: str | None,
        rag_enabled: bool,
        notes: str | None = None,
    ) -> int:
        """Insert a new eval_run row and return its auto-assigned primary key.

        Called once at the start of each batch evaluation job, before any per-sample inserts.
        The returned ``run_id`` is passed to ``insert_eval_results`` to associate result rows.

        Args:
            dataset:     Human-readable dataset name (e.g. ``"iam-line-test"``).
            n_samples:   Total number of samples that will be evaluated in this run.  Stored
                         upfront so the dashboard does not have to ``COUNT(*)`` results.
            model_ref:   Optional model checkpoint identifier for reproducibility tracking
                         (e.g. a RunPod job ID, a git SHA, or a HuggingFace revision string).
            rag_enabled: ``True`` if the RAG lexical corrector was active during inference.
            notes:       Optional free-text annotation, e.g. "trained on IAM train split only".

        Returns:
            The ``id`` (BIGINT) of the newly created ``eval_run`` row.
        """
        with self._connect() as conn, conn.cursor() as cur:
            # RETURNING id avoids a second round-trip to retrieve the auto-generated PK.
            cur.execute(
                f"INSERT INTO {config.EVAL_RUN_TABLE} "
                f"(dataset, n_samples, model_ref, rag_enabled, notes) "
                f"VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (dataset, n_samples, model_ref, rag_enabled, notes),
            )
            run_id: int = cur.fetchone()[0]  # fetchone() is safe: RETURNING always yields one row
            conn.commit()
            return run_id

    def list_eval_runs(self) -> list[dict]:
        """Return all eval_run rows ordered newest-first as a list of plain dicts.

        Used by the dashboard to populate the run selector and the history page.
        Column names are taken from ``cursor.description`` so this method is resilient to
        future column additions — no hardcoded positional unpacking.

        Returns:
            List of dicts with keys: ``id``, ``created_at``, ``dataset``, ``n_samples``,
            ``model_ref``, ``rag_enabled``.  Sorted descending by ``created_at``.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, created_at, dataset, n_samples, model_ref, rag_enabled "
                f"FROM {config.EVAL_RUN_TABLE} ORDER BY created_at DESC"
            )
            # Guard: if the schema has not been created yet (or if we're running against a stub
            # connection in tests), cur.description will be None and iterating it would raise a
            # TypeError.  An empty list is the safe, meaningful fallback.
            if cur.description is None:
                return []
            # Build column name list from cursor.description rather than hardcoding field order,
            # matching the pattern used in htr_sp3 API routes (dict(zip(cols, row))).
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def latest_run_id(self) -> int | None:
        """Return the id of the most recently created eval_run row, or None if the table is empty.

        Convenience helper for the dashboard's "show latest run" default state and for CLI tools
        that want to attach results to the most recent job without explicitly tracking the run_id.

        Returns:
            The ``id`` of the newest ``eval_run`` row, or ``None`` when the table is empty.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id FROM {config.EVAL_RUN_TABLE} ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            return row[0] if row else None

    # ------------------------------------------------------------------
    # eval_result bulk insert + aggregation
    # ------------------------------------------------------------------

    def insert_eval_results(self, run_id: int, rows: Iterable[EvalResultRow]) -> None:
        """Bulk-insert per-sample eval result rows into the eval_result table.

        Uses ``cursor.executemany`` for a single round-trip per batch.  The ``rows`` iterable is
        materialised into a list upfront so it can be passed to executemany (which requires a
        sequence, not a generator).

        The caller is responsible for passing the same ``run_id`` that was returned by
        ``create_eval_run``; no foreign-key lookup is performed here.

        Args:
            run_id: Primary key of the parent ``eval_run`` row.
            rows:   Iterable of ``EvalResultRow`` dataclasses (one per sample × scenario).
        """
        with self._connect() as conn, conn.cursor() as cur:
            # Materialise the iterable so executemany gets a sequence.
            # Map each dataclass to a tuple in the exact column order of the INSERT below.
            params = [
                (
                    run_id,
                    r.sample_id,
                    r.scenario,
                    r.text,
                    r.ground_truth,
                    r.cer,
                    r.wer,
                    r.latency_seconds,
                    r.log,
                    r.status_tag,
                )
                for r in rows
            ]
            cur.executemany(
                f"INSERT INTO {config.EVAL_RESULT_TABLE} "
                f"(run_id, sample_id, scenario, text, ground_truth, "
                f"cer, wer, latency_seconds, log, status_tag) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                params,
            )
            conn.commit()

    def eval_summary(self, run_id: int) -> list[dict]:
        """Return per-scenario aggregate stats for one run, computed in SQL.

        This is the primary data source for the dashboard's metric matrix table.  All aggregation
        is pushed to Postgres (``AVG``, ``COUNT``) so the Python layer stays thin and the query
        runs efficiently against the (run_id, scenario) index created in ``create_schema``.

        Returned values are rounded in Python (not SQL) to keep the DB query simple and to ensure
        consistent rounding behaviour regardless of Postgres version or platform.

        Args:
            run_id: Primary key of the eval_run whose results should be aggregated.

        Returns:
            List of dicts, one per scenario that has at least one result row.  Each dict contains:
                ``scenario``              — scenario slug (``"m1"`` … ``"m4"``).
                ``avg_cer``               — mean CER%, rounded to 2 dp; ``None`` if all NULL.
                ``avg_wer``               — mean WER%, rounded to 2 dp; ``None`` if all NULL.
                ``avg_latency_seconds``   — mean latency, rounded to 3 dp; ``None`` if all NULL.
                ``n``                     — number of result rows for this scenario (integer).
            Sorted alphabetically by ``scenario`` so the dashboard renders columns in a stable
            order (m1, m2, m3, m4).
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT scenario, "
                f"       AVG(cer)              AS avg_cer, "
                f"       AVG(wer)              AS avg_wer, "
                f"       AVG(latency_seconds)  AS avg_latency_seconds, "
                f"       COUNT(*)              AS n "
                f"FROM {config.EVAL_RESULT_TABLE} "
                f"WHERE run_id = %s "
                f"GROUP BY scenario "
                f"ORDER BY scenario",
                (run_id,),
            )
            # Guard: schema not yet created (or stub connection) yields cur.description == None.
            # Return an empty list rather than raising a TypeError.
            if cur.description is None:
                return []
            # Read column names from cur.description instead of positional tuple unpacking.
            # This is resilient to future SELECT additions — e.g. adding STDDEV(cer) AS stddev_cer
            # — because the dict is keyed by name, not by position.  Positional unpacking would
            # silently mis-assign values if a column were inserted before or between existing ones.
            cols = [c.name for c in cur.description]
            out = []
            for row in cur.fetchall():
                raw = dict(zip(cols, row))
                out.append({
                    "scenario":            raw["scenario"],
                    # Round in Python for predictability; AVG returns Decimal on some DBs.
                    "avg_cer":             None if raw["avg_cer"] is None else round(float(raw["avg_cer"]), 2),
                    "avg_wer":             None if raw["avg_wer"] is None else round(float(raw["avg_wer"]), 2),
                    "avg_latency_seconds": None if raw["avg_latency_seconds"] is None else round(float(raw["avg_latency_seconds"]), 3),
                    "n":                   int(raw["n"]),
                })
            return out

    # ------------------------------------------------------------------
    # upload_result CRUD
    # ------------------------------------------------------------------

    def insert_upload(
        self,
        filename: str,
        object_key: str,
        ground_truth: str | None,
        results: dict,
    ) -> int:
        """Insert a new upload_result row and return its auto-assigned primary key.

        Called by the single-image inference endpoint after the image has been uploaded to MinIO
        and the four-scenario inference has completed.  The ``results`` dict is serialised to JSON
        by this method before inserting into the ``JSONB`` column — callers pass a plain Python dict.

        Args:
            filename:     Original client-side filename (e.g. ``"receipt_001.png"``); display only.
            object_key:   MinIO object key used to generate presigned GET URLs later
                          (e.g. ``"uploads/3f2a1b.png"``).
            ground_truth: Optional reference transcription supplied by the user.
            results:      Folded results dict: ``{model: {text, cer, wer, latency_seconds, log,
                          status_tag}}``.  Stored verbatim as JSONB; use
                          ``htr_sp5.schemas.fold_results()`` to build this dict.

        Returns:
            The ``id`` (BIGINT) of the newly created ``upload_result`` row.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {config.UPLOAD_TABLE} "
                f"(filename, object_key, ground_truth, results) "
                f"VALUES (%s, %s, %s, %s) RETURNING id",
                (filename, object_key, ground_truth, json.dumps(results)),
            )
            up_id: int = cur.fetchone()[0]
            conn.commit()
            return up_id

    def list_uploads(self, limit: int, offset: int) -> list[dict]:
        """Return upload_result rows newest-first as a list of plain dicts, with pagination.

        Used by the dashboard history page.  The ``limit``/``offset`` pattern gives the frontend
        a simple cursor for infinite-scroll or numbered pagination without a database cursor.

        Args:
            limit:  Maximum number of rows to return.
            offset: Number of rows to skip from the start of the result set (0-indexed).

        Returns:
            List of dicts with keys: ``id``, ``created_at``, ``filename``, ``object_key``,
            ``ground_truth``, ``results``.  Sorted descending by ``created_at``.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, created_at, filename, object_key, ground_truth, results "
                f"FROM {config.UPLOAD_TABLE} "
                f"ORDER BY created_at DESC "
                f"LIMIT %s OFFSET %s",
                (limit, offset),
            )
            # Guard: a missing result set (e.g. schema not yet created) should yield an empty
            # list rather than a TypeError when iterating cur.description.
            if cur.description is None:
                return []
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_upload_object_key(self, upload_id: int) -> str | None:
        """Return the MinIO object key for a given upload_result row.

        Called by the ``/uploads/{id}/image`` presigned-URL endpoint in the FastAPI server.
        Fetching only the object_key column keeps the query minimal — the caller does not need
        the full row.

        Args:
            upload_id: Primary key of the ``upload_result`` row.

        Returns:
            The ``object_key`` string (e.g. ``"uploads/3f2a1b.png"``), or ``None`` if no row
            with that ``id`` exists.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT object_key FROM {config.UPLOAD_TABLE} WHERE id = %s",
                (upload_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None
