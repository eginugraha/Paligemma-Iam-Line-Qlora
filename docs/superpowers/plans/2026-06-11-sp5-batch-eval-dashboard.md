# SP-5 — Batch Eval, Dashboard & Upload History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist two result histories (offline batch-eval over IAM-test, and end-user uploads) in the existing pgvector Postgres, store uploaded images in MinIO, and add a Svelte `/dashboard` (FR-FE-05 matrix + Chart.js bar chart) and `/history` page.

**Architecture:** New `htr_sp5` package (config / schemas / store / objectstore / evalrun) mirroring the `htr_sp3` psycopg-v3 pattern. `scripts/eval_sp5.py` runs M1–M4 via `htr_sp2.orchestrator.detect_stream` over IAM-test and writes rows. `htr_sp2.api` gains a best-effort upload-persistence wrapper around the existing `/v1/detect` stream plus read-only endpoints. The SP-4 frontend gains a navbar layout, two routes, and a thin Chart.js wrapper.

**Tech Stack:** Python 3 / psycopg v3 / minio-py / FastAPI / pytest · SvelteKit + TypeScript / Chart.js / Vitest.

**Project conventions (apply to every task):**
- All code is **heavily commented** (thesis requirement — the author must be able to explain it). The code blocks below are functional; add explanatory comments/docstrings in the same density as the surrounding `htr_sp2`/`htr_sp3` files.
- Scenario identifiers are **`m1` / `m2` / `m3` / `m4`** — exactly the `model` field the orchestrator emits (NOT `m1_qlora`).
- Python tests live in `tests/` (flat dir, `test_sp5_*.py`), importable because `conftest.py` puts `src/` on `sys.path`.
- Real-DB tests are **opt-in** via `HTR_PG_TEST=1` (same `pytest.mark.skipif` guard as `tests/test_sp3_store.py`).
- Run backend tests with `python -m pytest`. Run frontend tests with `cd frontend && npm test`.

---

## File Structure

```
src/htr_sp5/
  __init__.py        # package marker
  config.py          # HTR_PG_DSN (reuse) + HTR_MINIO_* + table names
  schemas.py         # dataclasses + fold_results() pure helper (stream events -> results dict)
  store.py           # Sp5Store: psycopg v3 schema + insert/query (PgVectorStore pattern)
  objectstore.py     # MinioObjectStore: put_object + presigned_get_url (lazy minio import)
  evalrun.py         # run_eval(): pure-ish core used by the CLI (engine+store injected)
scripts/
  eval_sp5.py        # CLI: load IAM-test, build engine/corrector, call run_eval()
tests/
  test_sp5_config.py
  test_sp5_schemas.py
  test_sp5_store.py        # DB roundtrip gated by HTR_PG_TEST=1
  test_sp5_objectstore.py  # minio mocked
  test_sp5_evalrun.py      # FakeEngine + fake store
  test_sp5_api.py          # TestClient + monkeypatched store/objectstore
frontend/src/
  lib/types.ts             # MODIFY: add EvalRun, ScenarioSummary, UploadRecord
  lib/api.ts               # MODIFY: add fetchEvalRuns/fetchEvalSummary/fetchUploads/uploadImageUrl
  lib/api.dashboard.test.ts
  lib/BarChart.svelte      # Chart.js wrapper
  lib/BarChart.test.ts
  routes/+layout.svelte    # navbar
  routes/dashboard/+page.svelte
  routes/dashboard/page.test.ts
  routes/history/+page.svelte
  routes/history/page.test.ts
```

Modified backend files: `src/htr_sp2/api.py` (Task 7). Modified config/deps: `requirements-backend.txt`, `.env.example`, `frontend/package.json`.

---

## Task 1: SP-5 package + config

**Files:**
- Create: `src/htr_sp5/__init__.py`
- Create: `src/htr_sp5/config.py`
- Test: `tests/test_sp5_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp5_config.py
"""SP-5 config reads the shared Postgres DSN and the MinIO settings from the environment."""
import importlib


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import htr_sp5.config as cfg
    return importlib.reload(cfg)


def test_pg_dsn_defaults_and_table_names(monkeypatch):
    cfg = _reload(monkeypatch, HTR_PG_DSN="postgresql://u:p@h:5432/db")
    assert cfg.PG_DSN == "postgresql://u:p@h:5432/db"
    assert cfg.EVAL_RUN_TABLE == "eval_run"
    assert cfg.EVAL_RESULT_TABLE == "eval_result"
    assert cfg.UPLOAD_TABLE == "upload_result"


def test_minio_settings_from_env(monkeypatch):
    cfg = _reload(
        monkeypatch,
        HTR_MINIO_ENDPOINT="localhost:9000",
        HTR_MINIO_ACCESS_KEY="ak",
        HTR_MINIO_SECRET_KEY="sk",
        HTR_MINIO_BUCKET="htr-uploads",
        HTR_MINIO_SECURE="false",
    )
    assert cfg.MINIO_ENDPOINT == "localhost:9000"
    assert cfg.MINIO_ACCESS_KEY == "ak"
    assert cfg.MINIO_SECRET_KEY == "sk"
    assert cfg.MINIO_BUCKET == "htr-uploads"
    assert cfg.MINIO_SECURE is False


def test_minio_configured_flag(monkeypatch):
    cfg = _reload(monkeypatch, HTR_MINIO_ENDPOINT="", HTR_MINIO_ACCESS_KEY="", HTR_MINIO_SECRET_KEY="")
    assert cfg.minio_configured() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp5_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp5'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp5/__init__.py
"""SP-5: batch evaluation, statistics dashboard, and upload history."""
```

```python
# src/htr_sp5/config.py
"""Central configuration for SP-5 (history persistence + object storage).

Reuses the SP-3 Postgres DSN (HTR_PG_DSN) so both sub-projects share one database, and adds
the MinIO object-store settings used to persist end-user uploaded images.
"""
from __future__ import annotations

import os

# Load the repo-root .env the same way htr_sp3.config does, so credentials live in a gitignored
# file rather than being exported in every shell.
try:
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # parents[2] == repo root
except ImportError:
    pass

# --- Postgres (shared with SP-3) --------------------------------------------------------
PG_DSN = os.environ.get("HTR_PG_DSN", "postgresql://localhost:5432/htr")
EVAL_RUN_TABLE = "eval_run"
EVAL_RESULT_TABLE = "eval_result"
UPLOAD_TABLE = "upload_result"

# --- MinIO object storage (uploaded images) ---------------------------------------------
MINIO_ENDPOINT = os.environ.get("HTR_MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.environ.get("HTR_MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("HTR_MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.environ.get("HTR_MINIO_BUCKET", "htr-uploads")
# "true"/"false" string -> bool. Secure=True means the MinIO endpoint is behind TLS.
MINIO_SECURE = os.environ.get("HTR_MINIO_SECURE", "false").strip().lower() == "true"


def minio_configured() -> bool:
    """True only when endpoint + both keys are set, so the upload hook can no-op without MinIO."""
    return bool(MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp5_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp5/__init__.py src/htr_sp5/config.py tests/test_sp5_config.py
git commit -m "feat(sp5): config — shared PG DSN + MinIO settings"
```

---

## Task 2: Schemas + `fold_results` (stream events → results dict)

This is the pure logic that turns the NDJSON `result` events into the `{m1:{...}, m2:{...}}` dict stored in `upload_result.results` and used to build `eval_result` rows. No DB, fully unit-testable.

**Files:**
- Create: `src/htr_sp5/schemas.py`
- Test: `tests/test_sp5_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp5_schemas.py
from htr_sp5.schemas import fold_results, EvalResultRow, eval_rows_from_results


def _events():
    return [
        {"event": "meta", "filename": "a.png", "has_ground_truth": True},
        {"event": "result", "model": "m1", "text": "the cat", "cer": 5.0, "wer": 10.0,
         "latency_seconds": 0.7, "log": "Direct.", "status_tag": "Raw Output"},
        {"event": "error", "model": "m2", "message": "boom"},
        {"event": "result", "model": "m3", "text": "the cat", "cer": 0.0, "wer": 0.0,
         "latency_seconds": 1.1, "log": "RAG: ...", "status_tag": "Corrected"},
        {"event": "done"},
    ]


def test_fold_results_keeps_only_result_events_keyed_by_model():
    out = fold_results(_events())
    assert set(out.keys()) == {"m1", "m3"}            # error + meta + done dropped
    assert out["m1"]["text"] == "the cat"
    assert out["m1"]["cer"] == 5.0
    assert out["m3"]["status_tag"] == "Corrected"
    assert "event" not in out["m1"] and "model" not in out["m1"]


def test_eval_rows_from_results_emits_one_row_per_scenario():
    results = fold_results(_events())
    rows = eval_rows_from_results("sample-42", "the cat", results)
    assert {r.scenario for r in rows} == {"m1", "m3"}
    r1 = next(r for r in rows if r.scenario == "m1")
    assert isinstance(r1, EvalResultRow)
    assert r1.sample_id == "sample-42"
    assert r1.ground_truth == "the cat"
    assert r1.text == "the cat" and r1.cer == 5.0 and r1.latency_seconds == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp5_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp5.schemas'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp5/schemas.py
"""Plain dataclasses for SP-5 rows + pure helpers that fold NDJSON stream events into them.

Keeping this logic free of any DB/HTTP dependency makes the persistence path fully unit-testable:
the same `fold_results` output feeds both the upload history (JSONB blob) and the batch-eval rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# The six fields we persist per scenario (everything in a `result` event except the discriminator
# `event` and the `model` key, which becomes the dict key).
_RESULT_FIELDS = ("text", "cer", "wer", "latency_seconds", "log", "status_tag")


@dataclass(frozen=True)
class EvalResultRow:
    """One (sample x scenario) row destined for the eval_result table."""
    sample_id: str
    scenario: str
    text: str | None
    ground_truth: str | None
    cer: float | None
    wer: float | None
    latency_seconds: float | None
    log: str | None
    status_tag: str | None


def fold_results(events: Iterable[dict]) -> dict:
    """Reduce a sequence of NDJSON events to ``{model: {text,cer,wer,latency_seconds,log,status_tag}}``.

    Only ``result`` events are kept; ``meta``/``error``/``done`` are ignored. Errored scenarios
    simply do not appear (the dashboard/history render whatever scenarios succeeded).
    """
    out: dict[str, dict] = {}
    for evt in events:
        if evt.get("event") == "result":
            out[evt["model"]] = {k: evt.get(k) for k in _RESULT_FIELDS}
    return out


def eval_rows_from_results(sample_id: str, ground_truth: str | None, results: dict) -> list[EvalResultRow]:
    """Expand a folded results dict into one EvalResultRow per scenario for a batch-eval sample."""
    rows: list[EvalResultRow] = []
    for scenario, r in results.items():
        rows.append(EvalResultRow(
            sample_id=sample_id,
            scenario=scenario,
            text=r.get("text"),
            ground_truth=ground_truth,
            cer=r.get("cer"),
            wer=r.get("wer"),
            latency_seconds=r.get("latency_seconds"),
            log=r.get("log"),
            status_tag=r.get("status_tag"),
        ))
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp5_schemas.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp5/schemas.py tests/test_sp5_schemas.py
git commit -m "feat(sp5): schemas + fold_results (stream events -> rows)"
```

---

## Task 3: `Sp5Store` (Postgres persistence)

Mirrors `htr_sp3.store.PgVectorStore`: lazy `import psycopg`, `_connect()`, `with conn, conn.cursor()`. The real-DB roundtrip is opt-in (`HTR_PG_TEST=1`); without a DB we only test construction.

**Files:**
- Create: `src/htr_sp5/store.py`
- Test: `tests/test_sp5_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp5_store.py
"""Sp5Store persists eval runs/results and uploads in Postgres.

The roundtrip test hits a REAL database, so it is opt-in via HTR_PG_TEST=1 (same guard as
tests/test_sp3_store.py). Without it, we only check the store constructs from config.
"""
import os
import pytest

from htr_sp5.store import Sp5Store


def test_store_constructs_from_config_without_connecting():
    store = Sp5Store(dsn="postgresql://u:p@h:5432/db")
    assert store._dsn == "postgresql://u:p@h:5432/db"


@pytest.mark.skipif(
    not os.environ.get("HTR_PG_TEST"),
    reason="set HTR_PG_TEST=1 (with a live Postgres and HTR_PG_DSN) to run this test",
)
def test_eval_and_upload_roundtrip():
    from htr_sp5.schemas import EvalResultRow

    store = Sp5Store()
    store.create_schema()
    run_id = store.create_eval_run(dataset="iam-line-test", n_samples=1, model_ref="x", rag_enabled=True)
    store.insert_eval_results(run_id, [
        EvalResultRow("s1", "m1", "the cat", "the cat", 0.0, 0.0, 0.7, "Direct.", "Raw Output"),
    ])
    summary = store.eval_summary(run_id)
    assert summary[0]["scenario"] == "m1" and summary[0]["n"] == 1 and summary[0]["avg_cer"] == 0.0

    runs = store.list_eval_runs()
    assert runs[0]["id"] == run_id and runs[0]["n_samples"] == 1

    up_id = store.insert_upload(
        filename="a.png", object_key="uploads/a.png", ground_truth="the cat",
        results={"m1": {"text": "the cat", "cer": 0.0}},
    )
    uploads = store.list_uploads(limit=10, offset=0)
    assert uploads[0]["id"] == up_id and uploads[0]["object_key"] == "uploads/a.png"
    assert store.get_upload_object_key(up_id) == "uploads/a.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp5_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp5.store'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp5/store.py
"""PostgreSQL persistence for SP-5 (eval runs/results + upload history).

Same shape as htr_sp3.store.PgVectorStore: psycopg v3 is imported lazily so importing this module
never requires a DB driver; every method opens a short-lived connection via _connect().
"""
from __future__ import annotations

import json
from typing import Iterable

from htr_sp5 import config
from htr_sp5.schemas import EvalResultRow


class Sp5Store:
    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or config.PG_DSN

    def _connect(self):
        import psycopg
        return psycopg.connect(self._dsn)

    # --- schema -------------------------------------------------------------------------
    def create_schema(self) -> None:
        """Create the three SP-5 tables + the aggregation index (idempotent; no truncation)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.EVAL_RUN_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    dataset TEXT NOT NULL,
                    n_samples INT NOT NULL,
                    model_ref TEXT,
                    rag_enabled BOOLEAN NOT NULL,
                    notes TEXT
                )""")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.EVAL_RESULT_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    run_id BIGINT NOT NULL REFERENCES {config.EVAL_RUN_TABLE}(id) ON DELETE CASCADE,
                    sample_id TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    text TEXT,
                    ground_truth TEXT,
                    cer REAL,
                    wer REAL,
                    latency_seconds REAL,
                    log TEXT,
                    status_tag TEXT
                )""")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {config.EVAL_RESULT_TABLE}_run_scenario "
                f"ON {config.EVAL_RESULT_TABLE} (run_id, scenario)")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {config.UPLOAD_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    filename TEXT NOT NULL,
                    object_key TEXT NOT NULL,
                    ground_truth TEXT,
                    results JSONB NOT NULL
                )""")
            conn.commit()

    # --- batch eval ---------------------------------------------------------------------
    def create_eval_run(self, dataset: str, n_samples: int, model_ref: str | None,
                        rag_enabled: bool, notes: str | None = None) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {config.EVAL_RUN_TABLE} (dataset, n_samples, model_ref, rag_enabled, notes) "
                f"VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (dataset, n_samples, model_ref, rag_enabled, notes),
            )
            run_id = cur.fetchone()[0]
            conn.commit()
            return run_id

    def insert_eval_results(self, run_id: int, rows: Iterable[EvalResultRow]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {config.EVAL_RESULT_TABLE} "
                f"(run_id, sample_id, scenario, text, ground_truth, cer, wer, latency_seconds, log, status_tag) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(run_id, r.sample_id, r.scenario, r.text, r.ground_truth, r.cer, r.wer,
                  r.latency_seconds, r.log, r.status_tag) for r in rows],
            )
            conn.commit()

    def list_eval_runs(self) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, created_at, dataset, n_samples, model_ref, rag_enabled "
                f"FROM {config.EVAL_RUN_TABLE} ORDER BY created_at DESC")
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def latest_run_id(self) -> int | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {config.EVAL_RUN_TABLE} ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else None

    def eval_summary(self, run_id: int) -> list[dict]:
        """Per-scenario aggregate for one run, computed in SQL (the dashboard matrix data)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT scenario, AVG(cer) AS avg_cer, AVG(wer) AS avg_wer, "
                f"AVG(latency_seconds) AS avg_latency_seconds, COUNT(*) AS n "
                f"FROM {config.EVAL_RESULT_TABLE} WHERE run_id = %s GROUP BY scenario ORDER BY scenario",
                (run_id,),
            )
            out = []
            for scenario, avg_cer, avg_wer, avg_lat, n in cur.fetchall():
                out.append({
                    "scenario": scenario,
                    "avg_cer": None if avg_cer is None else round(float(avg_cer), 2),
                    "avg_wer": None if avg_wer is None else round(float(avg_wer), 2),
                    "avg_latency_seconds": None if avg_lat is None else round(float(avg_lat), 3),
                    "n": int(n),
                })
            return out

    # --- upload history -----------------------------------------------------------------
    def insert_upload(self, filename: str, object_key: str, ground_truth: str | None,
                    results: dict) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {config.UPLOAD_TABLE} (filename, object_key, ground_truth, results) "
                f"VALUES (%s, %s, %s, %s) RETURNING id",
                (filename, object_key, ground_truth, json.dumps(results)),
            )
            up_id = cur.fetchone()[0]
            conn.commit()
            return up_id

    def list_uploads(self, limit: int, offset: int) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, created_at, filename, object_key, ground_truth, results "
                f"FROM {config.UPLOAD_TABLE} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_upload_object_key(self, upload_id: int) -> str | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT object_key FROM {config.UPLOAD_TABLE} WHERE id = %s", (upload_id,))
            row = cur.fetchone()
            return row[0] if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp5_store.py -v`
Expected: PASS — `test_store_constructs_from_config_without_connecting` passes; the roundtrip is SKIPPED (no `HTR_PG_TEST`).

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp5/store.py tests/test_sp5_store.py
git commit -m "feat(sp5): Sp5Store — eval run/result + upload persistence (psycopg v3)"
```

---

## Task 4: `MinioObjectStore` (image upload + presigned URL)

**Files:**
- Create: `src/htr_sp5/objectstore.py`
- Test: `tests/test_sp5_objectstore.py`
- Modify: `requirements-backend.txt` (add `minio`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp5_objectstore.py
"""MinioObjectStore wraps minio-py. We inject a fake client so no real MinIO is needed."""
import io
from htr_sp5.objectstore import MinioObjectStore


class FakeMinio:
    def __init__(self):
        self.put_calls = []
        self.made_buckets = []
        self._exists = False

    def bucket_exists(self, bucket):
        return self._exists

    def make_bucket(self, bucket):
        self.made_buckets.append(bucket)
        self._exists = True

    def put_object(self, bucket, key, data, length, content_type=None):
        self.put_calls.append((bucket, key, data.read(), length, content_type))

    def presigned_get_object(self, bucket, key, expires=None):
        return f"http://minio/{bucket}/{key}?sig=abc"


def test_put_object_creates_bucket_then_uploads():
    fake = FakeMinio()
    store = MinioObjectStore(client=fake, bucket="htr-uploads")
    key = store.put_object("uploads/a.png", b"PNGBYTES", content_type="image/png")
    assert key == "uploads/a.png"
    assert fake.made_buckets == ["htr-uploads"]          # bucket didn't exist -> created
    assert fake.put_calls[0][0] == "htr-uploads"
    assert fake.put_calls[0][2] == b"PNGBYTES"
    assert fake.put_calls[0][3] == len(b"PNGBYTES")


def test_presigned_url_delegates_to_client():
    fake = FakeMinio()
    fake._exists = True
    store = MinioObjectStore(client=fake, bucket="htr-uploads")
    url = store.presigned_get_url("uploads/a.png")
    assert url == "http://minio/htr-uploads/uploads/a.png?sig=abc"


def test_new_object_key_has_uploads_prefix_and_extension():
    store = MinioObjectStore(client=FakeMinio(), bucket="b")
    key = store.new_object_key("My Photo.PNG")
    assert key.startswith("uploads/") and key.endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp5_objectstore.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp5.objectstore'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp5/objectstore.py
"""Thin wrapper over the MinIO (S3-compatible) client for storing uploaded images.

The minio package is imported lazily inside `from_config` so importing this module never requires
the dependency (tests inject a fake client). `put_object` is idempotent about the bucket: it
creates it on first use.
"""
from __future__ import annotations

import io
import os
import uuid
from datetime import timedelta

from htr_sp5 import config

# Map common image extensions to a normalized stored extension. Anything else falls back to .png.
_ALLOWED_EXT = {".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpg"}


class MinioObjectStore:
    def __init__(self, client, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def from_config(cls) -> "MinioObjectStore":
        """Build a store from env settings using the real minio client (lazy import)."""
        from minio import Minio

        client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE,
        )
        return cls(client=client, bucket=config.MINIO_BUCKET)

    def new_object_key(self, filename: str) -> str:
        """Build a collision-proof object key under uploads/, preserving a normalized extension."""
        ext = _ALLOWED_EXT.get(os.path.splitext(filename)[1].lower(), ".png")
        return f"uploads/{uuid.uuid4().hex}{ext}"

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def put_object(self, object_key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes under object_key (creating the bucket if needed); return the key."""
        self._ensure_bucket()
        self._client.put_object(
            self._bucket, object_key, io.BytesIO(data), length=len(data), content_type=content_type,
        )
        return object_key

    def presigned_get_url(self, object_key: str, expires_seconds: int = 3600) -> str:
        """Return a time-limited GET URL the browser can use directly for a thumbnail."""
        return self._client.presigned_get_object(
            self._bucket, object_key, expires=timedelta(seconds=expires_seconds),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp5_objectstore.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add the dependency**

Add to `requirements-backend.txt` under the SP-2 deps (before the `# Dev / test only:` line):

```
minio==7.2.7
```

- [ ] **Step 6: Commit**

```bash
git add src/htr_sp5/objectstore.py tests/test_sp5_objectstore.py requirements-backend.txt
git commit -m "feat(sp5): MinioObjectStore wrapper + minio dependency"
```

---

## Task 5: `run_eval` core + `scripts/eval_sp5.py` CLI

`run_eval` is the testable core (engine + corrector + store injected); the CLI just loads the IAM split and wires real objects.

**Files:**
- Create: `src/htr_sp5/evalrun.py`
- Create: `scripts/eval_sp5.py`
- Test: `tests/test_sp5_evalrun.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp5_evalrun.py
"""run_eval drives detect_stream over samples and writes rows via an injected store."""
from PIL import Image
from htr_sp5.evalrun import run_eval


class FakeEngine:
    """Minimal InferenceEngine: returns a fixed transcription regardless of input."""
    def run(self, image, prompt, max_new_tokens):
        return "the quick brown fox"


class RecordingStore:
    def __init__(self):
        self.run = None
        self.rows = []

    def create_eval_run(self, dataset, n_samples, model_ref, rag_enabled, notes=None):
        self.run = dict(dataset=dataset, n_samples=n_samples, model_ref=model_ref, rag_enabled=rag_enabled)
        return 1

    def insert_eval_results(self, run_id, rows):
        self.rows.extend(rows)


def _samples(n):
    img = Image.new("RGB", (8, 8), (255, 255, 255))
    return [{"sample_id": f"s{i}", "image": img, "ground_truth": "the quick brown fox"} for i in range(n)]


def test_run_eval_creates_run_and_inserts_rows_per_sample():
    store = RecordingStore()
    run_id = run_eval(_samples(2), FakeEngine(), corrector=None, store=store,
                    dataset="iam-line-test", model_ref="x")
    assert run_id == 1
    assert store.run["n_samples"] == 2 and store.run["rag_enabled"] is False
    # No corrector -> only m1 and m2 per sample -> 2 scenarios x 2 samples = 4 rows.
    assert len(store.rows) == 4
    assert {r.scenario for r in store.rows} == {"m1", "m2"}
    assert all(r.sample_id in {"s0", "s1"} for r in store.rows)
    assert next(r for r in store.rows if r.scenario == "m1").cer == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp5_evalrun.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp5.evalrun'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp5/evalrun.py
"""Batch-evaluation core: run M1-M4 over samples and persist eval rows.

Kept separate from the CLI so it can be tested with a FakeEngine and a recording store — no GPU,
no dataset download, no database. The CLI (scripts/eval_sp5.py) supplies the real objects.
"""
from __future__ import annotations

import json
from typing import Iterable

from htr_sp2.orchestrator import detect_stream
from htr_sp5.schemas import eval_rows_from_results, fold_results


def run_eval(samples: Iterable[dict], engine, corrector, store, *, dataset: str,
            model_ref: str | None) -> int:
    """Evaluate each sample through detect_stream and persist one eval_run + N*scenario rows.

    Each sample is a dict ``{"sample_id": str, "image": PIL.Image, "ground_truth": str|None}``.
    Returns the new run id. ``corrector`` is None to skip M3/M4, or an htr_sp3 RagCorrector.
    """
    samples = list(samples)
    run_id = store.create_eval_run(
        dataset=dataset, n_samples=len(samples), model_ref=model_ref,
        rag_enabled=corrector is not None,
    )
    for s in samples:
        # detect_stream yields NDJSON *strings*; parse them back to dicts to fold into results.
        events = [json.loads(line) for line in detect_stream(
            engine, s["image"], s["sample_id"], s.get("ground_truth"), corrector=corrector,
        )]
        results = fold_results(events)
        rows = eval_rows_from_results(s["sample_id"], s.get("ground_truth"), results)
        store.insert_eval_results(run_id, rows)
    return run_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp5_evalrun.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Write the CLI (no new test — it is thin glue over tested code)**

```python
# scripts/eval_sp5.py
#!/usr/bin/env python
"""SP-5 batch evaluation CLI: run M1-M4 over IAM-test and store results for the dashboard.

WHERE TO RUN
On a CUDA GPU machine (needs the real engine). Set ENGINE=runpod (+ its env) or run where the
local engine is available. Requires HTR_PG_DSN; use --rag to also evaluate M3/M4 (needs
HTR_ENABLE_RAG infra / pgvector ingested).

USAGE
    python scripts/eval_sp5.py --limit 200
    python scripts/eval_sp5.py --limit 200 --rag --model-ref eginugraha/paligemma-iam-line-qlora
"""
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htr_sp1 import data as iam_data            # noqa: E402
from htr_sp2.engine import get_engine           # noqa: E402
from htr_sp2.corrector_factory import get_corrector  # noqa: E402
from htr_sp5.evalrun import run_eval            # noqa: E402
from htr_sp5.store import Sp5Store              # noqa: E402


def _load_samples(limit: int):
    """Yield {sample_id, image, ground_truth} from the IAM test split, capped at `limit`."""
    splits = iam_data.load_iam_splits()
    test = splits["test"]
    out = []
    for i, rec in enumerate(test):
        if i >= limit:
            break
        out.append({
            "sample_id": str(rec.get("id", i)),
            "image": iam_data.ensure_rgb(rec["image"]),
            "ground_truth": rec["text"],
        })
    return out


def main():
    p = argparse.ArgumentParser(description="SP-5 batch evaluation over IAM-test.")
    p.add_argument("--limit", type=int, default=200, help="number of IAM-test samples to evaluate")
    p.add_argument("--rag", action="store_true", help="also evaluate M3/M4 (requires a corrector)")
    p.add_argument("--model-ref", default=None, help="model/adapter version label stored on the run")
    p.add_argument("--dataset", default="iam-line-test")
    args = p.parse_args()

    samples = _load_samples(args.limit)
    engine = get_engine()
    corrector = get_corrector() if args.rag else None
    store = Sp5Store()
    store.create_schema()
    run_id = run_eval(samples, engine, corrector, store,
                    dataset=args.dataset, model_ref=args.model_ref)
    print(f"eval_run {run_id} written: {len(samples)} samples, rag={corrector is not None}")


if __name__ == "__main__":
    main()
```

> NOTE for the implementer: verify the IAM record field names (`rec["image"]`, `rec["text"]`, `rec.get("id")`) against `htr_sp1/data.py` and an actual `load_iam_splits()` record; adjust the three keys in `_load_samples` if the dataset uses different column names. This is the only step that touches the real dataset shape.

- [ ] **Step 6: Run the full suite to confirm nothing regressed**

Run: `python -m pytest tests/test_sp5_evalrun.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/htr_sp5/evalrun.py scripts/eval_sp5.py tests/test_sp5_evalrun.py
git commit -m "feat(sp5): run_eval core + eval_sp5.py batch CLI"
```

---

## Task 6: API — upload persistence hook + read endpoints

Wrap the `/v1/detect` stream so that after it finishes, the image goes to MinIO and a row is inserted — best-effort (failures logged, never break the response). Add the read endpoints. Persistence is opt-in: if MinIO is not configured (or a store is not wired), the hook no-ops, so existing `tests/test_sp2_api.py` stays green.

**Files:**
- Modify: `src/htr_sp2/api.py`
- Test: `tests/test_sp5_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp5_api.py
"""SP-5 additions to the SP-2 API: upload persistence hook + read endpoints.

We monkeypatch the persistence dependencies so no MinIO/Postgres is needed.
"""
import io
import json
import importlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import htr_sp2.api as api_module


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


class RecordingStore:
    def __init__(self):
        self.uploads = []
        self.runs = [{"id": 7, "created_at": "2026-06-11T00:00:00Z", "dataset": "iam-line-test",
                    "n_samples": 2, "model_ref": "x", "rag_enabled": True}]
        self.summary = [{"scenario": "m1", "avg_cer": 5.0, "avg_wer": 10.0,
                        "avg_latency_seconds": 0.7, "n": 2}]

    def insert_upload(self, filename, object_key, ground_truth, results):
        self.uploads.append(dict(filename=filename, object_key=object_key,
                                ground_truth=ground_truth, results=results))
        return 1

    def list_eval_runs(self):
        return self.runs

    def latest_run_id(self):
        return 7

    def eval_summary(self, run_id):
        return self.summary

    def list_uploads(self, limit, offset):
        return [{"id": 1, "created_at": "2026-06-11T00:00:00Z", "filename": "a.png",
                "object_key": "uploads/a.png", "ground_truth": None,
                "results": {"m1": {"text": "hi"}}}]

    def get_upload_object_key(self, upload_id):
        return "uploads/a.png" if upload_id == 1 else None


class FakeObjectStore:
    def new_object_key(self, filename):
        return "uploads/fixed.png"

    def put_object(self, object_key, data, content_type="application/octet-stream"):
        return object_key

    def presigned_get_url(self, object_key, expires_seconds=3600):
        return f"http://minio/{object_key}"


@pytest.fixture
def client(monkeypatch):
    store = RecordingStore()
    objstore = FakeObjectStore()
    monkeypatch.setattr(api_module, "_get_store", lambda: store)
    monkeypatch.setattr(api_module, "_get_object_store", lambda: objstore)
    return TestClient(api_module.app), store


def test_detect_persists_upload_after_stream(client):
    c, store = client
    resp = c.post("/v1/detect", files={"file": ("a.png", _png(), "image/png")})
    assert resp.status_code == 200
    assert len(store.uploads) == 1
    up = store.uploads[0]
    assert up["object_key"] == "uploads/fixed.png"
    assert "m1" in up["results"]            # folded from the stream's result events


def test_eval_runs_and_summary_endpoints(client):
    c, _ = client
    assert c.get("/v1/eval/runs").json()[0]["id"] == 7
    summary = c.get("/v1/eval/summary").json()   # defaults to latest run
    assert summary[0]["scenario"] == "m1"


def test_uploads_list_and_image_redirect(client):
    c, _ = client
    assert c.get("/v1/uploads").json()[0]["filename"] == "a.png"
    r = c.get("/v1/uploads/1/image", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "http://minio/uploads/a.png"


def test_persistence_failure_does_not_break_stream(client, monkeypatch):
    c, store = client
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(store, "insert_upload", boom)
    resp = c.post("/v1/detect", files={"file": ("a.png", _png(), "image/png")})
    assert resp.status_code == 200                # stream still succeeds
    events = [json.loads(l) for l in resp.text.splitlines() if l.strip()]
    assert events[-1]["event"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp5_api.py -v`
Expected: FAIL — `AttributeError: module 'htr_sp2.api' has no attribute '_get_store'`.

- [ ] **Step 3: Modify `src/htr_sp2/api.py`**

Add imports near the existing imports:

```python
import json
import logging
from fastapi import Query
from fastapi.responses import RedirectResponse, JSONResponse

from htr_sp5 import config as sp5_config
from htr_sp5.schemas import fold_results

logger = logging.getLogger("htr_sp5")
```

Add lazy provider helpers (module level, after `app` is created). They return `None` when SP-5 isn't configured so persistence silently no-ops:

```python
def _get_store():
    """Return an Sp5Store, or None if Postgres isn't configured (persistence then no-ops)."""
    try:
        from htr_sp5.store import Sp5Store
        return Sp5Store()
    except Exception:  # pragma: no cover - import/config failure
        return None


def _get_object_store():
    """Return a MinioObjectStore, or None if MinIO isn't configured."""
    if not sp5_config.minio_configured():
        return None
    from htr_sp5.objectstore import MinioObjectStore
    return MinioObjectStore.from_config()
```

Replace the body of `detect` (from the `stream = detect_stream(...)` line through the `return`) with a wrapper that tees the stream and persists afterward:

```python
    corrector = get_corrector()
    stream = detect_stream(
        engine, image, file.filename or "upload", ground_truth, corrector=corrector
    )

    def _stream_and_persist():
        # Re-yield every NDJSON line to the client while collecting result events for history.
        events = []
        for line in stream:
            events.append(json.loads(line))
            yield line
        # Stream fully consumed -> persist best-effort. Never raise: the client already has
        # the full response, so a storage failure must not surface as a broken stream.
        try:
            store = _get_store()
            objstore = _get_object_store()
            if store is None or objstore is None:
                return
            object_key = objstore.new_object_key(file.filename or "upload.png")
            objstore.put_object(object_key, raw, content_type=file.content_type or "image/png")
            store.insert_upload(
                filename=file.filename or "upload",
                object_key=object_key,
                ground_truth=ground_truth,
                results=fold_results(events),
            )
        except Exception:  # pragma: no cover - exercised via the failure test
            logger.exception("SP-5 upload persistence failed (ignored)")

    return StreamingResponse(_stream_and_persist(), media_type="application/x-ndjson")
```

> NOTE: `raw` (the image bytes) is already read earlier in `detect` via `raw = await file.read()`. Keep that line; the wrapper closes over `raw`.

Add the read endpoints at the end of the file:

```python
@app.get("/v1/eval/runs")
def eval_runs():
    store = _get_store()
    return JSONResponse([] if store is None else store.list_eval_runs())


@app.get("/v1/eval/summary")
def eval_summary(run_id: int | None = Query(None)):
    store = _get_store()
    if store is None:
        return JSONResponse([])
    rid = run_id if run_id is not None else store.latest_run_id()
    return JSONResponse([] if rid is None else store.eval_summary(rid))


@app.get("/v1/uploads")
def uploads(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    store = _get_store()
    return JSONResponse([] if store is None else store.list_uploads(limit, offset))


@app.get("/v1/uploads/{upload_id}/image")
def upload_image(upload_id: int):
    store = _get_store()
    objstore = _get_object_store()
    if store is None or objstore is None:
        raise HTTPException(status_code=404, detail="object storage not configured")
    key = store.get_upload_object_key(upload_id)
    if key is None:
        raise HTTPException(status_code=404, detail="upload not found")
    return RedirectResponse(objstore.presigned_get_url(key))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sp5_api.py tests/test_sp2_api.py -v`
Expected: PASS — all SP-5 API tests pass AND the four existing SP-2 API tests still pass (persistence no-ops there because `_get_object_store()` returns None without MinIO config).

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/api.py tests/test_sp5_api.py
git commit -m "feat(sp5): /v1/detect upload-persistence hook + eval/uploads endpoints"
```

---

## Task 7: Frontend — types + API client functions

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Test: `frontend/src/lib/api.dashboard.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/lib/api.dashboard.test.ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { fetchEvalRuns, fetchEvalSummary, fetchUploads, uploadImageUrl } from './api';

afterEach(() => vi.restoreAllMocks());

function mockFetchJson(payload: unknown) {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => payload }) as Response));
}

describe('dashboard/history api', () => {
  it('fetchEvalRuns returns parsed runs', async () => {
    mockFetchJson([{ id: 7, dataset: 'iam-line-test', n_samples: 2, model_ref: 'x', rag_enabled: true,
      created_at: '2026-06-11T00:00:00Z' }]);
    const runs = await fetchEvalRuns('http://api');
    expect(runs[0].id).toBe(7);
    expect(fetch).toHaveBeenCalledWith('http://api/v1/eval/runs');
  });

  it('fetchEvalSummary passes run_id when given', async () => {
    mockFetchJson([{ scenario: 'm1', avg_cer: 5, avg_wer: 10, avg_latency_seconds: 0.7, n: 2 }]);
    const s = await fetchEvalSummary(7, 'http://api');
    expect(s[0].scenario).toBe('m1');
    expect(fetch).toHaveBeenCalledWith('http://api/v1/eval/summary?run_id=7');
  });

  it('fetchEvalSummary omits run_id when null', async () => {
    mockFetchJson([]);
    await fetchEvalSummary(null, 'http://api');
    expect(fetch).toHaveBeenCalledWith('http://api/v1/eval/summary');
  });

  it('fetchUploads builds pagination query', async () => {
    mockFetchJson([{ id: 1, filename: 'a.png', object_key: 'uploads/a.png', ground_truth: null,
      results: {}, created_at: '2026-06-11T00:00:00Z' }]);
    const u = await fetchUploads(20, 0, 'http://api');
    expect(u[0].filename).toBe('a.png');
    expect(fetch).toHaveBeenCalledWith('http://api/v1/uploads?limit=20&offset=0');
  });

  it('uploadImageUrl builds the image endpoint URL', () => {
    expect(uploadImageUrl(1, 'http://api')).toBe('http://api/v1/uploads/1/image');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/lib/api.dashboard.test.ts`
Expected: FAIL — `fetchEvalRuns` is not exported by `./api`.

- [ ] **Step 3: Add types to `frontend/src/lib/types.ts`**

Append:

```typescript
/** A batch-evaluation run (one row of eval_run). */
export interface EvalRun {
  id: number;
  created_at: string;
  dataset: string;
  n_samples: number;
  model_ref: string | null;
  rag_enabled: boolean;
}

/** Per-scenario aggregate for one run (a row of the dashboard matrix). */
export interface ScenarioSummary {
  scenario: ScenarioId;
  avg_cer: number | null;
  avg_wer: number | null;
  avg_latency_seconds: number | null;
  n: number;
}

/** One end-user upload history record. */
export interface UploadRecord {
  id: number;
  created_at: string;
  filename: string;
  object_key: string;
  ground_truth: string | null;
  results: Record<string, { text?: string; cer?: number | null; wer?: number | null;
    latency_seconds?: number | null; log?: string; status_tag?: string }>;
}
```

- [ ] **Step 4: Add functions to `frontend/src/lib/api.ts`**

Add (reuse the existing `DEFAULT_BASE` constant):

```typescript
import type { EvalRun, ScenarioSummary, UploadRecord } from './types';

export async function fetchEvalRuns(baseUrl: string = DEFAULT_BASE): Promise<EvalRun[]> {
  const res = await fetch(`${baseUrl}/v1/eval/runs`);
  if (!res.ok) throw new Error(`eval/runs failed: HTTP ${res.status}`);
  return (await res.json()) as EvalRun[];
}

export async function fetchEvalSummary(
  runId: number | null,
  baseUrl: string = DEFAULT_BASE
): Promise<ScenarioSummary[]> {
  const q = runId == null ? '' : `?run_id=${runId}`;
  const res = await fetch(`${baseUrl}/v1/eval/summary${q}`);
  if (!res.ok) throw new Error(`eval/summary failed: HTTP ${res.status}`);
  return (await res.json()) as ScenarioSummary[];
}

export async function fetchUploads(
  limit = 50,
  offset = 0,
  baseUrl: string = DEFAULT_BASE
): Promise<UploadRecord[]> {
  const res = await fetch(`${baseUrl}/v1/uploads?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`uploads failed: HTTP ${res.status}`);
  return (await res.json()) as UploadRecord[];
}

export function uploadImageUrl(uploadId: number, baseUrl: string = DEFAULT_BASE): string {
  return `${baseUrl}/v1/uploads/${uploadId}/image`;
}
```

- [ ] **Step 5: Run test + typecheck**

Run: `cd frontend && npm test -- src/lib/api.dashboard.test.ts && npm run check`
Expected: PASS (5 tests) and `npm run check` reports 0 errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/lib/api.dashboard.test.ts
git commit -m "feat(sp5): frontend types + eval/uploads API client functions"
```

---

## Task 8: Frontend — navbar layout + Chart.js `BarChart.svelte`

**Files:**
- Create: `frontend/src/routes/+layout.svelte`
- Create: `frontend/src/lib/BarChart.svelte`
- Test: `frontend/src/lib/BarChart.test.ts`
- Modify: `frontend/package.json` (add `chart.js`)

- [ ] **Step 1: Install Chart.js**

Run: `cd frontend && npm install chart.js@4`
Expected: `chart.js` added to `dependencies` in `package.json`.

- [ ] **Step 2: Write the failing test**

```typescript
// frontend/src/lib/BarChart.test.ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render } from '@testing-library/svelte';

// Mock chart.js so the test asserts the wrapper's lifecycle, not canvas pixels.
const destroy = vi.fn();
const update = vi.fn();
const ChartMock = vi.fn(() => ({ destroy, update, data: { labels: [], datasets: [] } }));
vi.mock('chart.js/auto', () => ({ default: ChartMock }));

import BarChart from './BarChart.svelte';

afterEach(() => { vi.clearAllMocks(); });

describe('BarChart.svelte', () => {
  const labels = ['M1', 'M2', 'M3', 'M4'];
  const datasets = [
    { label: 'Avg CER', data: [17, 16, 5, 5] },
    { label: 'Avg WER', data: [28, 27, 10, 9] }
  ];

  it('constructs a Chart with the passed labels and datasets', () => {
    render(BarChart, { props: { labels, datasets } });
    expect(ChartMock).toHaveBeenCalledTimes(1);
    const cfg = ChartMock.mock.calls[0][1];
    expect(cfg.type).toBe('bar');
    expect(cfg.data.labels).toEqual(labels);
    expect(cfg.data.datasets[0].label).toBe('Avg CER');
  });

  it('destroys the chart on unmount', () => {
    const { unmount } = render(BarChart, { props: { labels, datasets } });
    unmount();
    expect(destroy).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd frontend && npm test -- src/lib/BarChart.test.ts`
Expected: FAIL — cannot find `./BarChart.svelte`.

- [ ] **Step 4: Write `frontend/src/lib/BarChart.svelte`**

```svelte
<script lang="ts">
  // Thin Chart.js wrapper. `chart.js/auto` auto-registers the bar controller/scales so we don't
  // hand-register components. The chart is created after mount (needs the <canvas>) and destroyed
  // on unmount; it is recreated when `labels`/`datasets` change.
  import { onMount, onDestroy } from 'svelte';
  import Chart from 'chart.js/auto';

  export let labels: string[] = [];
  export let datasets: { label: string; data: number[] }[] = [];

  let canvas: HTMLCanvasElement;
  let chart: Chart | undefined;

  function build() {
    if (!canvas) return;
    chart?.destroy();
    chart = new Chart(canvas, {
      type: 'bar',
      data: { labels, datasets },
      options: { responsive: true, scales: { y: { beginAtZero: true } } }
    });
  }

  onMount(build);
  // Rebuild when inputs change (Svelte 5 reactive statement re-runs build on dep change after mount).
  $: if (chart) { labels; datasets; build(); }
  onDestroy(() => chart?.destroy());
</script>

<canvas bind:this={canvas} role="img" aria-label="Scenario comparison bar chart"></canvas>
```

- [ ] **Step 5: Write `frontend/src/routes/+layout.svelte`**

```svelte
<script lang="ts">
  // App shell: a persistent top navbar across all routes. SvelteKit renders the active page
  // into <slot/>. Plain links keep this dependency-free (consistent with SP-4's no-UI-lib rule).
  import { page } from '$app/stores';
  const links = [
    { href: '/', label: 'Detect' },
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/history', label: 'History' }
  ];
</script>

<nav>
  {#each links as l}
    <a href={l.href} class:active={$page.url.pathname === l.href}>{l.label}</a>
  {/each}
</nav>
<main><slot /></main>

<style>
  nav { display: flex; gap: 1rem; padding: 0.75rem 1rem; border-bottom: 1px solid #ddd; }
  nav a { text-decoration: none; color: #444; font-weight: 600; }
  nav a.active { color: #1a73e8; border-bottom: 2px solid #1a73e8; }
  main { padding: 1rem; }
</style>
```

- [ ] **Step 6: Run test + checks**

Run: `cd frontend && npm test -- src/lib/BarChart.test.ts && npm run check`
Expected: PASS (2 tests), `npm run check` 0 errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/BarChart.svelte frontend/src/lib/BarChart.test.ts \
        frontend/src/routes/+layout.svelte frontend/package.json frontend/package-lock.json
git commit -m "feat(sp5): navbar layout + Chart.js BarChart wrapper"
```

---

## Task 9: Frontend — `/dashboard` page (matrix + chart)

**Files:**
- Create: `frontend/src/routes/dashboard/+page.svelte`
- Test: `frontend/src/routes/dashboard/page.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/routes/dashboard/page.test.ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/svelte';

vi.mock('$lib/BarChart.svelte', async () => {
  // Replace the chart with a stub so the page test doesn't depend on Chart.js/canvas.
  const Stub = (await import('../../lib/__stubs__/Empty.svelte')).default;
  return { default: Stub };
});

vi.mock('$lib/api', () => ({
  fetchEvalRuns: vi.fn(async () => [
    { id: 7, created_at: '2026-06-11T00:00:00Z', dataset: 'iam-line-test', n_samples: 2,
      model_ref: 'x', rag_enabled: true }
  ]),
  fetchEvalSummary: vi.fn(async () => [
    { scenario: 'm1', avg_cer: 17.4, avg_wer: 28.3, avg_latency_seconds: 0.78, n: 2 },
    { scenario: 'm3', avg_cer: 5.0, avg_wer: 10.0, avg_latency_seconds: 1.1, n: 2 }
  ])
}));

import Dashboard from './+page.svelte';

afterEach(() => vi.clearAllMocks());

describe('/dashboard', () => {
  it('renders one matrix row per scenario from the summary', async () => {
    render(Dashboard);
    await waitFor(() => expect(screen.getByText('17.4%')).toBeInTheDocument());
    expect(screen.getByText('28.3%')).toBeInTheDocument();
    expect(screen.getByText('5%')).toBeInTheDocument();        // m3 avg_cer
    // 2 scenario rows -> 2 latency cells
    expect(screen.getByText('0.78 s')).toBeInTheDocument();
  });
});
```

Also create the shared stub used above:

```svelte
<!-- frontend/src/lib/__stubs__/Empty.svelte -->
<div data-testid="stub"></div>
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/routes/dashboard/page.test.ts`
Expected: FAIL — cannot find `./+page.svelte`.

- [ ] **Step 3: Write `frontend/src/routes/dashboard/+page.svelte`**

```svelte
<script lang="ts">
  // FR-FE-05 dashboard: a run selector, a per-scenario matrix table, and a CER/WER bar chart.
  import { onMount } from 'svelte';
  import { fetchEvalRuns, fetchEvalSummary } from '$lib/api';
  import type { EvalRun, ScenarioSummary } from '$lib/types';
  import BarChart from '$lib/BarChart.svelte';

  const SCENARIO_LABEL: Record<string, string> = {
    m1: 'M1 QLoRA', m2: 'M2 +CoT', m3: 'M3 +RAG', m4: 'M4 Hybrid'
  };

  let runs: EvalRun[] = [];
  let selectedRunId: number | null = null;
  let summary: ScenarioSummary[] = [];
  let error = '';

  const pct = (v: number | null) => (v == null ? '—' : `${v}%`);
  const secs = (v: number | null) => (v == null ? '—' : `${v} s`);

  async function loadSummary() {
    summary = await fetchEvalSummary(selectedRunId);
  }

  onMount(async () => {
    try {
      runs = await fetchEvalRuns();
      selectedRunId = runs.length ? runs[0].id : null;   // newest first
      await loadSummary();
    } catch (e) {
      error = (e as Error).message;
    }
  });

  // Chart inputs derived from the same summary that feeds the table.
  $: labels = summary.map((s) => SCENARIO_LABEL[s.scenario] ?? s.scenario);
  $: datasets = [
    { label: 'Avg CER', data: summary.map((s) => s.avg_cer ?? 0) },
    { label: 'Avg WER', data: summary.map((s) => s.avg_wer ?? 0) }
  ];
</script>

<h1>Global Statistics (Batch Evaluation)</h1>

{#if error}<p class="err">{error}</p>{/if}

{#if runs.length}
  <label>
    Run:
    <select bind:value={selectedRunId} on:change={loadSummary}>
      {#each runs as r}
        <option value={r.id}>
          {new Date(r.created_at).toLocaleString()} · {r.dataset} · n={r.n_samples}
          · RAG {r.rag_enabled ? 'on' : 'off'}
        </option>
      {/each}
    </select>
  </label>
{:else}
  <p>No evaluation runs yet. Run <code>python scripts/eval_sp5.py</code> first.</p>
{/if}

{#if summary.length}
  <table>
    <thead>
      <tr><th>Scenario</th><th>Avg CER</th><th>Avg WER</th><th>Avg Latency</th><th>N</th></tr>
    </thead>
    <tbody>
      {#each summary as s}
        <tr>
          <td>{SCENARIO_LABEL[s.scenario] ?? s.scenario}</td>
          <td>{pct(s.avg_cer)}</td>
          <td>{pct(s.avg_wer)}</td>
          <td>{secs(s.avg_latency_seconds)}</td>
          <td>{s.n}</td>
        </tr>
      {/each}
    </tbody>
  </table>

  <div class="chart"><BarChart {labels} {datasets} /></div>
{/if}

<style>
  table { border-collapse: collapse; margin-top: 1rem; }
  th, td { border: 1px solid #ddd; padding: 0.4rem 0.8rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .chart { max-width: 640px; margin-top: 1.5rem; }
  .err { color: #c00; }
</style>
```

- [ ] **Step 4: Run test + checks**

Run: `cd frontend && npm test -- src/routes/dashboard/page.test.ts && npm run check`
Expected: PASS (1 test), `npm run check` 0 errors.

> NOTE: `5.0%` renders as `5%` because the number is interpolated directly. The test asserts `5%`; if you choose to format with fixed decimals, update the test to match.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/routes/dashboard frontend/src/lib/__stubs__/Empty.svelte
git commit -m "feat(sp5): /dashboard page — scenario matrix + CER/WER bar chart"
```

---

## Task 10: Frontend — `/history` page (upload list + expand + thumbnail)

**Files:**
- Create: `frontend/src/routes/history/+page.svelte`
- Test: `frontend/src/routes/history/page.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/routes/history/page.test.ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';

vi.mock('$lib/api', () => ({
  fetchUploads: vi.fn(async () => [
    { id: 1, created_at: '2026-06-11T00:00:00Z', filename: 'a.png', object_key: 'uploads/a.png',
      ground_truth: 'the cat',
      results: { m1: { text: 'the cat', cer: 0, wer: 0, latency_seconds: 0.7, log: 'Direct.',
        status_tag: 'Raw Output' } } }
  ]),
  uploadImageUrl: (id: number) => `http://api/v1/uploads/${id}/image`
}));

import History from './+page.svelte';

afterEach(() => vi.clearAllMocks());

describe('/history', () => {
  it('lists uploads with a thumbnail and expands detail on click', async () => {
    render(History);
    await waitFor(() => expect(screen.getByText('a.png')).toBeInTheDocument());
    const img = screen.getByRole('img') as HTMLImageElement;
    expect(img.src).toBe('http://api/v1/uploads/1/image');
    // detail (log) hidden until the row is expanded
    expect(screen.queryByText('Direct.')).not.toBeInTheDocument();
    await fireEvent.click(screen.getByText('a.png'));
    expect(screen.getByText('Direct.')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/routes/history/page.test.ts`
Expected: FAIL — cannot find `./+page.svelte`.

- [ ] **Step 3: Write `frontend/src/routes/history/+page.svelte`**

```svelte
<script lang="ts">
  // Upload history: newest-first list of end-user uploads. Each row shows a thumbnail + filename;
  // clicking a row toggles a detail panel with the full per-scenario parameters (SP-4 fields).
  import { onMount } from 'svelte';
  import { fetchUploads, uploadImageUrl } from '$lib/api';
  import type { UploadRecord } from '$lib/types';

  const SCENARIO_LABEL: Record<string, string> = {
    m1: 'M1 QLoRA', m2: 'M2 +CoT', m3: 'M3 +RAG', m4: 'M4 Hybrid'
  };

  let uploads: UploadRecord[] = [];
  let expanded: number | null = null;
  let error = '';

  const toggle = (id: number) => (expanded = expanded === id ? null : id);
  const pct = (v: number | null | undefined) => (v == null ? '—' : `${v}%`);

  onMount(async () => {
    try {
      uploads = await fetchUploads();
    } catch (e) {
      error = (e as Error).message;
    }
  });
</script>

<h1>Upload History</h1>

{#if error}<p class="err">{error}</p>{/if}
{#if !uploads.length}<p>No uploads yet.</p>{/if}

<ul>
  {#each uploads as u}
    <li>
      <button class="row" on:click={() => toggle(u.id)}>
        <img src={uploadImageUrl(u.id)} alt={u.filename} width="64" height="32" />
        <span class="name">{u.filename}</span>
        <span class="time">{new Date(u.created_at).toLocaleString()}</span>
      </button>

      {#if expanded === u.id}
        <div class="detail">
          {#each Object.entries(u.results) as [scenario, r]}
            <div class="scenario">
              <strong>{SCENARIO_LABEL[scenario] ?? scenario}</strong>
              <div>Text: {r.text ?? '—'}</div>
              <div>CER: {pct(r.cer)} · WER: {pct(r.wer)} · {r.latency_seconds ?? '—'} s</div>
              <div>Status: {r.status_tag ?? '—'}</div>
              <div class="log">{r.log ?? ''}</div>
            </div>
          {/each}
        </div>
      {/if}
    </li>
  {/each}
</ul>

<style>
  ul { list-style: none; padding: 0; }
  li { border-bottom: 1px solid #eee; }
  .row { display: flex; align-items: center; gap: 1rem; width: 100%; background: none;
    border: none; padding: 0.6rem 0; cursor: pointer; text-align: left; }
  .row img { object-fit: cover; border: 1px solid #ccc; }
  .name { font-weight: 600; }
  .time { color: #888; font-size: 0.85rem; margin-left: auto; }
  .detail { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem; padding: 0.5rem 0 1rem 5rem; }
  .scenario { font-size: 0.9rem; }
  .log { color: #666; font-size: 0.8rem; margin-top: 0.25rem; }
  .err { color: #c00; }
</style>
```

- [ ] **Step 4: Run test + checks**

Run: `cd frontend && npm test -- src/routes/history/page.test.ts && npm run check`
Expected: PASS (1 test), `npm run check` 0 errors.

- [ ] **Step 5: Run the entire frontend + backend suites**

Run: `cd frontend && npm test && npm run build`
Expected: all Vitest tests pass; production build succeeds.
Run: `cd .. && python -m pytest`
Expected: all backend tests pass (SP-5 DB roundtrip SKIPPED without `HTR_PG_TEST`).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/routes/history
git commit -m "feat(sp5): /history page — upload list with thumbnail + expandable detail"
```

---

## Task 11: Wire-up docs (.env.example) + manual e2e smoke

**Files:**
- Modify: `.env.example`
- Create: `README-sp5.md` (brief, matching README-sp2/3 style)

- [ ] **Step 1: Document the new env keys in `.env.example`**

Append a SP-5 section:

```
# --- SP-5: MinIO object storage for uploaded images (src/htr_sp5/config.py) ---
#   Leave unset to disable upload persistence (the /v1/detect stream still works).
HTR_MINIO_ENDPOINT=localhost:9000
HTR_MINIO_ACCESS_KEY=minioadmin
HTR_MINIO_SECRET_KEY=minioadmin
HTR_MINIO_BUCKET=htr-uploads
HTR_MINIO_SECURE=false
```

- [ ] **Step 2: Write `README-sp5.md`** with: what SP-5 adds (two histories, dashboard), how to create the schema (`python -c "import sys; sys.path.insert(0,'src'); from htr_sp5.store import Sp5Store; Sp5Store().create_schema()"`), how to run a batch eval (`python scripts/eval_sp5.py --limit 50`), and how to view (`/dashboard`, `/history`). Keep it ~40 lines, same tone as `README-sp3.md`.

- [ ] **Step 3: Commit**

```bash
git add .env.example README-sp5.md
git commit -m "docs(sp5): document MinIO env keys + SP-5 usage"
```

- [ ] **Step 4: Manual e2e smoke (record results, do not automate)**

Prereqs: a running MinIO, Postgres reachable via `HTR_PG_DSN`, the trained engine available.

1. Create the schema (Step 2 one-liner) and a MinIO bucket (auto-created on first upload).
2. `HTR_ENABLE_RAG=1 uvicorn htr_sp2.api:app --app-dir src`
3. `cd frontend && npm run dev`; open the app, upload a handwriting PNG, confirm M1–M4 stream in.
4. Open `/history` → the upload appears with a working thumbnail and expandable detail.
5. `python scripts/eval_sp5.py --limit 5` → open `/dashboard` → matrix + bar chart render.

Expected: histories populate; dashboard shows aggregated CER/WER per scenario.

---

## Self-Review notes (already reconciled)

- **Spec coverage:** two histories (Tasks 3,5,6) · MinIO image storage (Task 4,6) · offline batch CLI (Task 5) · auto upload-persist after stream (Task 6) · `/dashboard` matrix + Chart.js (Tasks 8,9) · `/history` (Task 10) · presigned thumbnail (Tasks 4,6) · navbar 2 routes (Task 8) · env keys (Task 1,11). FR-FE-05 = Task 9.
- **Scenario naming:** `m1`–`m4` everywhere (matches orchestrator `model`), not `m1_qlora`.
- **Backward compat:** Task 6 hook no-ops without MinIO config, so existing `tests/test_sp2_api.py` stays green.
- **Type consistency:** `fold_results` (Task 2) feeds both `eval_rows_from_results` (Task 5) and the upload `results` JSONB (Task 6); `ScenarioSummary`/`EvalRun`/`UploadRecord` (Task 7) match the store query shapes (Task 3).
