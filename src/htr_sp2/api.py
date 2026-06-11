"""FastAPI surface for the SP-2 backend.

POST /v1/detect takes a multipart image (+ optional ground_truth) and streams NDJSON
results. The image is validated up front so a bad upload fails with 422 BEFORE the stream
opens; engine failures mid-stream are reported per-model by the orchestrator instead.

Thesis context (SP-2, Chapter 4 — System Design)
-------------------------------------------------
This module is the HTTP interface that connects the SP-4 frontend (Svelte) and the
SP-5 batch evaluator to the SP-2 inference backend. It deliberately keeps HTTP
concerns out of the orchestrator layer so the orchestrator remains testable without
an HTTP server.

Key design decisions
--------------------
1. Early image validation (``image.load()``)
   FastAPI's ``StreamingResponse`` writes the status line (200 OK) and headers
   immediately, before the generator starts yielding. Once that happens the client
   has committed to reading a 200 stream — we can no longer send a 422 error.
   Calling ``image.load()`` forces PIL to fully decode the image bytes *before* we
   create the StreamingResponse; if decoding fails, we raise HTTPException(422) and
   the client receives a proper error JSON rather than a truncated 200 stream.

2. Sync generator → StreamingResponse
   ``detect_stream`` is a synchronous generator (uses ``yield``, not ``async yield``).
   FastAPI's ``StreamingResponse`` accepts both sync and async iterables, so wrapping
   it directly works without an ``asyncio.run_in_executor`` adapter. The tradeoff is
   that a slow engine blocks the event loop for the duration of each ``engine.run``
   call; for the thesis prototype this is acceptable since we run one request at a time.
   A production version would wrap the sync generator in ``run_in_executor``.

3. Content-type: application/x-ndjson
   The IANA media type for newline-delimited JSON is ``application/x-ndjson``.
   Using this instead of ``application/json`` signals to clients (and the SP-5
   evaluator) that the body is a stream of independent JSON objects, one per line,
   rather than a single JSON value.
"""
from __future__ import annotations

import io
import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

# get_engine reads config.ENGINE (defaults to "fake") and returns the appropriate
# InferenceEngine implementation. Called inside the request handler so that
# monkeypatching config.ENGINE in tests works correctly.
from htr_sp2.engine import get_engine

# detect_stream is the core orchestration function. It accepts a pre-decoded PIL
# Image (not raw bytes) so this API layer owns all HTTP/multipart concerns and the
# orchestrator stays HTTP-agnostic.
from htr_sp2.orchestrator import detect_stream

from htr_sp2 import config
from htr_sp2.corrector_factory import get_corrector

# SP-5: config module exposes ``minio_configured()`` which guards all MinIO calls.
# Imported here (not inside the helper functions) so that monkeypatching
# ``sp5_config.minio_configured`` in tests works without reloading the module.
from htr_sp5 import config as sp5_config

# SP-5: ``fold_results`` reduces a list of NDJSON event dicts into the compact
# ``{model: {text, cer, wer, …}}`` dict stored as a JSONB blob in upload_result.
from htr_sp5.schemas import fold_results

# Named logger for all SP-5 persistence messages.  Using a dedicated logger
# (instead of the root logger) lets operators filter SP-5 noise independently
# of SP-2 inference logs.
logger = logging.getLogger("htr_sp5")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HTR SP-2 Backend",
    version="0.1.0",
    description=(
        "SP-2 inference backend for the Handwritten Text Recognition thesis. "
        "Streams NDJSON results from two PaliGemma inference scenarios "
        "(M1 baseline and M2 chain-of-thought) for a single uploaded image."
    ),
)

# CORS so the SP-4 browser frontend (a different origin, e.g. the Vite dev server) can call
# the API. Credentials are not used, so a wildcard origin is safe. Origins come from config
# (env HTR_CORS_ORIGINS, default "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# SP-5 lazy provider helpers
# ---------------------------------------------------------------------------
# These two functions are intentionally module-level *callables* rather than
# module-level *values*.  The distinction matters for two reasons:
#
# 1. Test monkeypatching: pytest's ``monkeypatch.setattr(api_module, "_get_store", ...)``
#    replaces the *name* in the module's namespace.  If we stored the store instance in a
#    module-level variable at import time, tests would need to patch the variable AND reset
#    it after each test.  With a callable, ``monkeypatch.setattr`` is the only hook needed,
#    and pytest automatically undoes it after each test.
#
# 2. Graceful degradation (no-op without configuration): when MinIO / Postgres are not
#    configured (which is always the case in CI and local dev without Docker), the functions
#    return None.  The persistence hook in ``_stream_and_persist`` checks for None and
#    exits early, so the existing SP-2 tests (which do not configure MinIO) are unaffected.
#    This is the critical backward-compatibility guarantee: the SP-5 persistence layer is
#    entirely invisible to the SP-2 API tests.


def _get_store():
    """Return an ``Sp5Store`` instance, or ``None`` if the store module cannot be imported.

    ``Sp5Store.__init__`` only stores the DSN string — it does **not** open a database
    connection.  The connection is lazy: it is established the first time a query method
    is called (e.g. ``insert_upload``).  Therefore this function almost always returns a
    valid ``Sp5Store`` object; it returns ``None`` only in the rare case where importing
    ``htr_sp5.store`` itself raises an exception (e.g. a broken install or a missing
    dependency at import time).

    The real backward-compatibility safety net is two-fold:
      1. ``_get_object_store()`` returns ``None`` when MinIO credentials are absent, and
         the ``if store is None or objstore is None: return`` guard in ``_stream_and_persist``
         skips all persistence when either provider is unavailable.
      2. The ``except Exception`` in ``_stream_and_persist`` catches and logs any DB error
         that surfaces at query time (e.g. a Postgres outage at ``insert_upload``), so a
         storage failure can never corrupt the client's already-delivered response.

    The ``store is None`` branch of the guard is therefore defensive-only — it protects
    against an import failure, not against a missing ``HTR_PG_DSN`` or unreachable DB.

    Design note — no module-level caching:
        We do NOT cache the returned instance in a module-level variable because
        ``monkeypatch.setattr`` needs to replace this *function* itself, not a cached
        attribute of a previously constructed object.  A new ``Sp5Store`` is constructed
        per request; for the thesis prototype the overhead is negligible since no
        connection is opened here.  A production version would use a connection pool.

    Returns:
        ``Sp5Store`` instance (DB connection is lazy), or ``None`` if the module import fails.
    """
    try:
        from htr_sp5.store import Sp5Store
        return Sp5Store()
    except Exception:  # pragma: no cover - import failure path (extremely rare)
        return None


def _get_object_store():
    """Return a ``MinioObjectStore`` instance, or ``None`` when MinIO is not configured.

    ``sp5_config.minio_configured()`` checks that all three MinIO credentials
    (endpoint, access key, secret key) are non-empty strings.  If any are missing we
    return ``None`` immediately, avoiding a failed TCP connection attempt to a
    non-existent MinIO endpoint.

    Design note — guard before import:
        The ``MinioObjectStore`` import lives *inside* this function so that the ``minio``
        Python package is only required when MinIO is actually configured.  This keeps the
        SP-2 API importable on machines where ``minio`` is not installed (e.g. a minimal
        inference-only deployment that doesn't need upload history).

    Returns:
        ``MinioObjectStore`` if MinIO credentials are fully set, ``None`` otherwise.
    """
    if not sp5_config.minio_configured():
        return None
    from htr_sp5.objectstore import MinioObjectStore
    return MinioObjectStore.from_config()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness probe used by Docker HEALTHCHECK and Kubernetes readiness probes.

    Returns a JSON body ``{"status": "ok"}`` with HTTP 200. No database or GPU
    state is checked here; the intent is purely to verify the process is alive
    and the HTTP server is accepting connections.

    Returns:
        dict: ``{"status": "ok"}``
    """
    return {"status": "ok"}


@app.post("/v1/detect")
async def detect(
    file: UploadFile = File(...),
    ground_truth: str | None = Form(None),
):
    """Run M1 (baseline) and M2 (CoT) on the uploaded image and stream NDJSON results.

    This endpoint is the primary interface consumed by the SP-4 Svelte frontend and
    the SP-5 batch evaluator. It accepts a multipart/form-data POST with:

    - ``file``         — the handwriting image (PNG, JPEG, or any PIL-supported format).
    - ``ground_truth`` — (optional) the human-verified transcription. When supplied,
                         CER and WER are computed for each scenario; when omitted,
                         ``cer`` and ``wer`` fields in result events are JSON null.

    The response is an ``application/x-ndjson`` stream of JSON objects, one per line.
    Stream structure:
        1. ``meta``   — announces the filename and whether ground truth is available.
        2. ``result`` — one per scenario (m1, m2); carries text, metrics, timing.
           OR ``error`` — if a scenario failed; remaining scenarios still run.
        3. ``done``   — signals stream end.

    Error handling:
        - 422 Unprocessable Entity — returned synchronously (before streaming begins)
          if the uploaded file is not a decodable image (corrupt bytes, wrong format).
          HTTPException with this status is safe because the response has not started yet.
        - Per-scenario engine failures are reported as ``error`` events inside the
          stream (not HTTP errors) because the stream has already started by then.

    Args:
        file:         Uploaded image file (multipart UploadFile from FastAPI).
        ground_truth: Optional reference transcription for metric computation.

    Returns:
        StreamingResponse with media_type ``"application/x-ndjson"``.

    Raises:
        HTTPException(422): If the uploaded bytes cannot be decoded as an image.
    """
    # --- 1. Read the raw bytes from the multipart upload ----------------------
    # UploadFile is an async wrapper; we must await .read() to get the bytes.
    raw = await file.read()

    # --- 2. Validate and fully decode the image up front ----------------------
    # PIL.Image.open() is lazy: it reads the header but defers pixel decoding.
    # Calling image.load() forces full decoding now. If the bytes are corrupt or
    # not a recognised image format, PIL raises one of the exceptions below —
    # and we can still return HTTP 422 because no streaming has started yet.
    #
    # Without image.load(), a corrupt image would only fail later, inside the
    # StreamingResponse generator. At that point the HTTP status line (200 OK)
    # and content-type header have already been flushed to the client, so we
    # cannot change the status code. The client would receive a truncated 200
    # stream with no indication of the real error.
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force decode now so a bad image fails here, not mid-stream
    except (UnidentifiedImageError, OSError, ValueError):
        # UnidentifiedImageError — PIL cannot identify the file format.
        # OSError             — covers truncated files and format-level decode errors.
        # ValueError          — rare; raised by some PIL plugins for out-of-range data.
        raise HTTPException(status_code=422, detail="invalid or undecodable image")

    # --- 3. Build the engine and orchestrate the inference --------------------
    # get_engine() returns FakeEngine by default (config.ENGINE == "fake").
    # In production, set the ENGINE env var to "runpod".
    engine = get_engine()

    # get_corrector() returns None unless HTR_ENABLE_RAG is set (then a cached PgVector-backed
    # RagCorrector). When present, detect_stream additionally emits m3 (corrected m1) and m4
    # (corrected m2); when None, the stream is M1/M2 only — unchanged behaviour.
    corrector = get_corrector()
    stream = detect_stream(
        engine, image, file.filename or "upload", ground_truth, corrector=corrector
    )

    # --- 4. Wrap the stream with SP-5 upload-persistence (tee approach) ------
    # We need to accomplish two things at once:
    #   (a) Yield every NDJSON line to the HTTP client as before (unchanged SP-2 behaviour).
    #   (b) After the stream is fully consumed, persist the image and results to MinIO/Postgres.
    #
    # The challenge is that ``StreamingResponse`` starts flushing to the client as soon as the
    # generator yields its first line, so we cannot do post-processing *after* the response
    # object is returned.  The solution is a "tee" generator: it re-yields each line verbatim
    # (preserving the streaming UX) while collecting parsed events in a local list.  Only when
    # the inner ``stream`` is exhausted — meaning the client has received all lines — does the
    # generator execute the persistence logic.
    #
    # Why best-effort (try/except that logs and swallows)?
    #   At the point persistence runs, the HTTP response body has already been written.
    #   If we were to raise an exception here, FastAPI would produce a truncated body with no
    #   error status change (the 200 status line was already sent).  The client would receive a
    #   partial stream with no indication of what went wrong.  It is strictly better to swallow
    #   the exception, log it for operator investigation, and leave the client with a complete
    #   and valid response.  Missing a history record is a non-critical loss; corrupting the
    #   response would break every SP-4 frontend consumer.
    #
    # Closure over ``raw``:
    #   ``raw`` (the bytes read at the top of this handler) is captured by the nested generator
    #   closure.  We use ``raw`` rather than re-reading ``file`` because the UploadFile async
    #   cursor is already at EOF after the first ``await file.read()``.

    def _stream_and_persist():
        """Tee the detect stream to the client, then persist the upload best-effort.

        Yields every NDJSON line from the underlying ``detect_stream`` generator while
        simultaneously collecting parsed event dicts.  After the inner generator is
        exhausted, attempts to store the image in MinIO and record the upload in
        Postgres.  Any exception during persistence is logged and silently ignored so
        that a storage outage cannot corrupt the client's already-delivered response.

        Yields:
            bytes-or-str: Each NDJSON line from ``detect_stream``, unmodified.
        """
        # Accumulate parsed event dicts so we can fold them into the results JSONB blob
        # after the stream ends.  We parse each line as it passes through rather than
        # storing raw lines, because ``fold_results`` expects dicts, not strings.
        events: list[dict] = []

        for line in stream:
            # Parse the line now (while the generator is hot) and buffer it.
            # json.loads is cheap compared to inference latency, so this adds no
            # meaningful overhead to the streaming path.
            events.append(json.loads(line))
            # Re-yield the original line (not the re-serialised dict) to preserve
            # byte-for-byte fidelity with what detect_stream produced.
            yield line

        # --- Persistence (best-effort) -----------------------------------------
        # The entire stream has been consumed and delivered to the client.
        # We now attempt to save the image and its results.  Any failure here is
        # explicitly swallowed: the client already has a complete 200 response and
        # there is no mechanism to signal an error after streaming has ended.
        try:
            store = _get_store()
            objstore = _get_object_store()

            # If either provider returns None (e.g. MinIO/Postgres not configured in
            # this deployment), skip persistence entirely.  This is the no-op path
            # that keeps the existing SP-2 CI tests green without any MinIO setup.
            if store is None or objstore is None:
                return

            # Allocate a unique object key for this upload, then store the raw bytes.
            # ``new_object_key`` generates a timestamped UUID path under "uploads/";
            # ``put_object`` performs the actual MinIO PUT request.
            object_key = objstore.new_object_key(file.filename or "upload.png")
            objstore.put_object(
                object_key,
                raw,  # captured from the enclosing request handler's ``raw = await file.read()``
                content_type=file.content_type or "image/png",
            )

            # Fold the accumulated stream events into the compact results dict and
            # write one row to the upload_result table.
            # ``fold_results`` filters out meta/error/done events, keeping only
            # "result" events, and maps them to {model: {text, cer, wer, …}}.
            store.insert_upload(
                filename=file.filename or "upload",
                object_key=object_key,
                ground_truth=ground_truth,
                results=fold_results(events),
            )

        except Exception:
            # Log the full traceback so operators can investigate storage issues,
            # but do NOT re-raise: the response is already complete for the client.
            # Swallowing this exception is intentional — see the module-level comment
            # on _stream_and_persist for the full rationale.
            logger.exception("SP-5 upload persistence failed (ignored)")

    # --- 5. Return the wrapped streaming response -----------------------------
    # The generator replaces the bare ``stream`` from SP-2 while keeping the same
    # media_type and streaming semantics.  From the client's perspective nothing has
    # changed: they still receive the same sequence of NDJSON lines.
    return StreamingResponse(_stream_and_persist(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# SP-5 read endpoints — dashboard and history views
# ---------------------------------------------------------------------------
# These are thin read-only wrappers around the SP-5 store.  They deliberately
# contain NO business logic beyond querying the store and serialising the result:
# the store (``Sp5Store``) owns query construction and the schemas module owns
# data shape; this layer only handles HTTP concerns (status codes, JSON serialisation,
# query parameter parsing).
#
# All four endpoints call ``_get_store()`` / ``_get_object_store()`` for the same
# reason as the ``/v1/detect`` persistence hook: the functions are the monkeypatch
# seam for tests, and they return None when storage is not configured so that the
# endpoints degrade gracefully (returning [] / 404) instead of crashing.


@app.get(
    "/v1/eval/runs",
    summary="List all batch evaluation runs",
    response_description="Array of eval_run rows, newest first.",
)
def eval_runs():
    """Return a JSON array of all batch evaluation runs recorded in the database.

    Each element corresponds to one row in the ``eval_run`` table, representing a
    single batch job that processed a dataset through the M1–M4 scenarios.  The
    SP-5 dashboard uses this list to populate the run-selector dropdown.

    When the database is not configured (e.g. in local dev without Postgres),
    the store is None and we return an empty array rather than a 500 error, so the
    dashboard renders an empty state instead of crashing.

    Returns:
        JSONResponse: Array of eval-run dicts (may be empty if no runs exist or
                      storage is not configured).
    """
    store = _get_store()
    # Guard: return [] rather than crashing when Postgres is unavailable.
    return JSONResponse([] if store is None else store.list_eval_runs())


@app.get(
    "/v1/eval/summary",
    summary="Per-scenario aggregate metrics for an evaluation run",
    response_description="Array of per-scenario aggregate metric rows.",
)
def eval_summary(run_id: int | None = Query(None)):
    """Return per-scenario aggregate metrics (avg CER, avg WER, avg latency) for a run.

    When ``run_id`` is omitted the endpoint defaults to the most recently created run
    (via ``store.latest_run_id()``).  This is the most common use case: the dashboard
    home page always shows the latest evaluation without requiring the user to specify
    a run ID.

    Args:
        run_id: Optional query parameter.  When provided, return metrics for that
                specific run; when absent, default to the latest run id from the store.

    Returns:
        JSONResponse: Array of per-scenario aggregate dicts, or [] if no data is
                      available (unconfigured storage, unknown run_id, or no results yet).
    """
    store = _get_store()
    if store is None:
        return JSONResponse([])

    # Resolve the run id: use the explicit query param, or fall back to latest.
    # ``latest_run_id()`` may itself return None if the eval_run table is empty,
    # in which case we also return [] to avoid passing None to eval_summary().
    rid = run_id if run_id is not None else store.latest_run_id()
    return JSONResponse([] if rid is None else store.eval_summary(rid))


@app.get(
    "/v1/uploads",
    summary="Paginated upload history",
    response_description="Page of upload_result rows.",
)
def uploads(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Return a paginated list of user uploads from the upload_result table.

    Used by the SP-5 dashboard's history page to show every image that was ever
    submitted to ``/v1/detect``.  Pagination is cursor-free (limit/offset) because
    the thesis demo dataset is small and total ordering by insertion time is fine.

    Args:
        limit:  Maximum rows to return (1–200, default 50).  Capped at 200 to prevent
                accidental full-table scans from browser clients.
        offset: Number of rows to skip before returning results (default 0, first page).

    Returns:
        JSONResponse: Array of upload_result dicts, or [] if storage is not configured.
    """
    store = _get_store()
    return JSONResponse([] if store is None else store.list_uploads(limit, offset))


@app.get(
    "/v1/uploads/{upload_id}/image",
    summary="Redirect to the presigned MinIO URL for an uploaded image",
    response_description="307 Temporary Redirect to a time-limited presigned URL.",
)
def upload_image(upload_id: int):
    """Redirect the caller to a presigned MinIO URL for the requested uploaded image.

    Rather than proxying the image bytes through this API server (which would double
    the bandwidth and add latency), we generate a short-lived presigned URL that lets
    the browser fetch the image directly from MinIO.  This is the standard pattern for
    S3-compatible object storage: the API server issues the presigned URL, and the client
    fetches the object from MinIO in a separate request.

    ``RedirectResponse`` defaults to status 307 (Temporary Redirect), which is correct
    here: the presigned URL is ephemeral (it expires after ``expires_seconds``), so
    caching the redirect destination would cause stale links.  Status 302 would have the
    same expiry semantics but some clients incorrectly cache it; 307 is safer.

    Args:
        upload_id: Primary key of the ``upload_result`` row.

    Returns:
        RedirectResponse (307): Location header set to the presigned MinIO URL.

    Raises:
        HTTPException(404): If object storage is not configured, or if no upload with
                            the given ``upload_id`` exists in the database.
    """
    store = _get_store()
    objstore = _get_object_store()

    # Both store and objstore are required: the store looks up the object key, and
    # objstore generates the presigned URL.  If either is unavailable we cannot serve
    # the image, so we return 404 rather than 500 to avoid alarming the browser.
    if store is None or objstore is None:
        raise HTTPException(status_code=404, detail="object storage not configured")

    key = store.get_upload_object_key(upload_id)
    if key is None:
        # The upload_id does not exist in the database.
        raise HTTPException(status_code=404, detail="upload not found")

    # Generate a presigned GET URL valid for 1 hour (default in MinioObjectStore).
    # RedirectResponse defaults to 307 — intentionally kept so clients do not cache the URL.
    return RedirectResponse(objstore.presigned_get_url(key))
