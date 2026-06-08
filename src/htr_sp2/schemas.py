"""Builders for the NDJSON events streamed by /v1/detect.

Plain dicts (not pydantic) keep it trivial to serialize with json.dumps and to assert on
in tests. These functions are the ONE place the event shape is defined.

NDJSON streaming contract
--------------------------
The /v1/detect endpoint emits one JSON object per line (newline-delimited JSON, NDJSON).
Each object has a mandatory ``"event"`` discriminator field. The sequence is:

    1. ``meta``   — one per request; announces filename and whether ground-truth is known.
    2. ``result`` — one per scenario (up to four); carries text, metrics, and timing.
       OR ``error`` — if that scenario failed; stream continues with remaining scenarios.
    3. ``done``   — exactly one, final line; signals the stream is closed.

Consumers (SP-4 frontend, SP-5 batch evaluator) switch on ``event`` to route each line
to the right handler. Using string discriminators rather than HTTP status codes lets us
carry partial results: even if two of four scenarios fail, the client still receives the
two successful results before the ``done`` line.

Why plain dicts?
-----------------
Pydantic would give us runtime validation and IDE type hints, but it also adds a
dependency, requires .model_dump() before json.dumps, and makes test assertions more
verbose. Since the shapes are tiny and fixed, plain dicts are simpler and the tests
directly assert equality without conversion overhead.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------

def meta_event(filename: str, has_ground_truth: bool) -> dict:
    """Build the opening event announcing what image is about to be processed.

    Emitted once per /v1/detect request, before any ``result`` or ``error``
    events. Lets the frontend display the filename and decide whether to show
    the CER/WER columns (only meaningful when ground truth exists).

    Args:
        filename:         The original filename of the uploaded image
                          (e.g. "line_01.png"). Used for display only.
        has_ground_truth: True when the caller supplied a reference string;
                          False when only the image was provided.

    Returns:
        Dict: ``{"event": "meta", "filename": ..., "has_ground_truth": ...}``
    """
    return {
        "event": "meta",
        "filename": filename,
        "has_ground_truth": has_ground_truth,
    }


def result_event(
    model: str,
    text: str,
    cer: float | None,
    wer: float | None,
    latency_seconds: float,
    log: str,
    status_tag: str,
) -> dict:
    """Build a result event carrying one scenario's transcription and metrics.

    Emitted once per successfully completed scenario. There are four scenarios
    in the thesis comparison:
      - Baseline (PaliGemma, no CoT)
      - CoT (PaliGemma + chain-of-thought prompt)
      - RAG (PaliGemma + pgvector retrieval)
      - CoT+RAG (combined)

    Args:
        model:            Short scenario identifier (e.g. ``"baseline"``,
                          ``"cot"``, ``"rag"``, ``"cot_rag"``).
        text:             The raw transcription produced by the model.
        cer:              Character Error Rate as a percentage (0–100), or
                          ``None`` when no ground truth was supplied.
        wer:              Word Error Rate as a percentage (0–100), or ``None``
                          when no ground truth was supplied.
        latency_seconds:  Wall-clock seconds from request start to response
                          received. Used for the thesis performance analysis.
        log:              A short human-readable status string (e.g.
                          ``"Inference OK"``). Useful for debugging in the UI.
        status_tag:       One of a fixed set of UI tags that the frontend maps
                          to a badge colour (e.g. ``"Raw Output"``,
                          ``"CoT Output"``).

    Returns:
        Dict with ``"event": "result"`` and all the above fields.
    """
    return {
        "event": "result",
        "model": model,
        "text": text,
        "cer": cer,           # None signals "no ground truth" to the frontend
        "wer": wer,           # None signals "no ground truth" to the frontend
        "latency_seconds": latency_seconds,
        "log": log,
        "status_tag": status_tag,
    }


def error_event(model: str, message: str) -> dict:
    """Build an error event for a scenario that raised an exception.

    The orchestrator catches per-scenario exceptions and emits this event
    instead of crashing the entire stream. The remaining scenarios still run.

    Args:
        model:   The scenario identifier that failed (matches the ``model``
                 field in result_event).
        message: The exception message or a short human-readable description
                 of the failure.

    Returns:
        Dict: ``{"event": "error", "model": ..., "message": ...}``
    """
    return {
        "event": "error",
        "model": model,
        "message": message,
    }


def done_event() -> dict:
    """Build the terminal event that closes the NDJSON stream.

    Emitted exactly once, as the final line. The frontend uses this to hide
    any loading spinner and enable the export button. Batch evaluators use it
    to know the response body is fully consumed.

    Returns:
        Dict: ``{"event": "done"}``
    """
    return {"event": "done"}
