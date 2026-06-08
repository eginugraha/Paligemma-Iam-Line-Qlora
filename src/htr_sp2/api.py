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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

# get_engine reads config.ENGINE (defaults to "fake") and returns the appropriate
# InferenceEngine implementation. Called inside the request handler so that
# monkeypatching config.ENGINE in tests works correctly.
from htr_sp2.engine import get_engine

# detect_stream is the core orchestration function. It accepts a pre-decoded PIL
# Image (not raw bytes) so this API layer owns all HTTP/multipart concerns and the
# orchestrator stays HTTP-agnostic.
from htr_sp2.orchestrator import detect_stream

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

    # detect_stream is a synchronous generator; it yields one NDJSON line string
    # per event (meta, result×N, done). StreamingResponse consumes the generator
    # and flushes each yielded chunk to the client as it arrives.
    stream = detect_stream(engine, image, file.filename or "upload", ground_truth)

    # --- 4. Return the streaming response -------------------------------------
    # media_type="application/x-ndjson" tells clients the body is newline-delimited
    # JSON rather than a single JSON document. The SP-4 frontend and SP-5 evaluator
    # both check this content-type to select the correct parser.
    return StreamingResponse(stream, media_type="application/x-ndjson")
