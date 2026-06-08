"""The detect flow: run each scenario against the engine and yield NDJSON lines.

M1 is a direct transcription; M2 reuses the same model with the CoT prompt and its output
is parsed into (text, reasoning). CER/WER reuse htr_sp1.metrics. One failing scenario emits
an error event and the stream continues to the next — a dead column never kills the others.

Thesis context (SP-2, Chapter 4 — System Design)
-------------------------------------------------
SP-2 compares two inference strategies (scenarios) on the same fine-tuned PaliGemma model:

  M1 — Baseline: The plain transcription prompt from SP-1. The model emits the answer
       directly, so the raw output *is* the transcription. Simple, fast, low token usage.

  M2 — Chain-of-Thought (CoT): An extended prompt that asks the model to reason briefly
       about ambiguous strokes before committing to a final answer. The output contains a
       reasoning section followed by "Final: <answer>". ``parse_cot`` extracts these parts.

This module wires the two scenarios into a single NDJSON stream so the frontend (SP-4) and
batch evaluator (SP-5) see a uniform interface regardless of how many scenarios are active.

NDJSON stream structure (one JSON object per line)
--------------------------------------------------
  1. ``meta``   — opens the stream; announces filename and ground-truth availability.
  2. ``result`` — one per successful scenario, in scenario order (m1, m2).
     OR ``error`` — one per failed scenario; the stream continues with the next scenario.
  3. ``done``   — final line; signals the stream is fully consumed.

The generator pattern (yield) lets FastAPI (SP-3) wrap this in a ``StreamingResponse``
without buffering all results in memory — important when later scenarios are added (RAG, etc.)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

# htr_sp1.metrics provides CER/WER computed via jiwer. Reusing SP-1 functions ensures the
# metric definitions are identical across all scenarios and all sub-projects (SP-1, SP-2,
# SP-5), which is critical for the thesis comparison tables to be meaningful.
from htr_sp1.metrics import cer as cer_metric
from htr_sp1.metrics import wer as wer_metric

# config holds the prompt strings and token caps for each scenario so they are in one
# place. Changing a prompt here automatically propagates to both the orchestrator and any
# batch or evaluation script that imports config.
from htr_sp2 import config, schemas

# parse_cot splits the M2 raw generation into (final_answer, reasoning_log). It handles
# both the happy path (model followed the 'Final:' format) and the fallback (model did not).
from htr_sp2.cot import parse_cot

# EngineError is the single exception type that engine implementations raise for any
# failure (network timeout, bad HTTP status, garbled JSON). The orchestrator catches it
# per scenario so one failing model does not abort the whole stream.
from htr_sp2.engine import EngineError, InferenceEngine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed log message for M1. M1 has no reasoning phase, so the log is a short description
# of what happened. Kept as a module-level constant so tests can assert on it exactly.
M1_LOG = "Direct visual token translation completed."


# ---------------------------------------------------------------------------
# Internal model spec descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelSpec:
    """Everything that differs between M1 and M2, collected in one place.

    Using a frozen dataclass (rather than ad-hoc dicts) makes it impossible to
    accidentally mutate a spec at runtime and clarifies each field's purpose.

    Attributes:
        model:          Short scenario identifier used in result/error events
                        (e.g. "m1", "m2"). Matches the ``model`` field the
                        frontend switches on.
        prompt:         Fully-rendered prompt string passed to engine.run. M1
                        uses the plain SP-1 transcription prompt; M2 uses the
                        CoT prompt from cot.py.
        max_new_tokens: Hard token cap. M1 uses the SP-1 cap (~64 tokens for
                        typical IAM lines). M2 uses 256 to accommodate the
                        reasoning prefix before 'Final:'.
        status_tag:     Badge label shown in the frontend table column header
                        (e.g. "Raw Output", "Reasoned"). Mapped from config
                        so all label strings are in one file.
    """
    model: str
    prompt: str
    max_new_tokens: int
    status_tag: str


# Ordered list of scenario specs. The stream emits events in this order, which determines
# the column order in the SP-4 frontend table. M1 first, M2 second — matching the thesis
# baseline-then-enhanced structure used throughout the comparison chapters.
_SPECS = [
    _ModelSpec("m1", config.M1_PROMPT, config.M1_MAX_NEW_TOKENS, config.M1_STATUS_TAG),
    _ModelSpec("m2", config.M2_PROMPT, config.M2_MAX_NEW_TOKENS, config.M2_STATUS_TAG),
]


# ---------------------------------------------------------------------------
# NDJSON serialisation helper
# ---------------------------------------------------------------------------

def _line(event: dict) -> str:
    """Serialise one event dict as a single NDJSON line (JSON + newline).

    NDJSON (Newline-Delimited JSON) is the wire format for the /v1/detect
    stream. Each line is a self-contained JSON object so consumers can parse
    events incrementally using a simple ``readline()`` loop without waiting
    for the full body.

    Args:
        event: A plain dict built by one of the ``schemas.*_event`` builders.

    Returns:
        The JSON-encoded string followed by exactly one newline character.
    """
    return json.dumps(event) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_stream(
    engine: InferenceEngine,
    image: Image.Image,
    filename: str,
    ground_truth: str | None,
) -> Iterator[str]:
    """Run M1 then M2 against *engine* and yield NDJSON lines.

    This is the heart of the SP-2 backend. It is a synchronous generator that
    the FastAPI layer wraps in a ``StreamingResponse``. Each ``yield`` pushes
    one complete JSON line to the client without waiting for all scenarios to
    finish — the frontend can start rendering M1's column while M2 is still
    running.

    Failure isolation
    -----------------
    If ``engine.run`` raises ``EngineError`` for a given scenario, an
    ``error`` event is emitted for that scenario and the loop advances to the
    next scenario. This means:
      - A dead M1 (e.g. bad prompt format) does not block M2.
      - The ``done`` event is always emitted, so the client is never left
        waiting for a stream that silently stalled.

    Metrics
    -------
    CER and WER are computed via ``htr_sp1.metrics`` (jiwer). They are only
    computed when *ground_truth* is not None. Both are rounded to two decimal
    places before serialisation so the JSON is human-readable and float
    rounding artefacts are suppressed.

    Args:
        engine:       Any object satisfying ``InferenceEngine`` (the Protocol).
                      In production this is ``RunPodEngine``; in tests it is
                      ``FakeEngine``.
        image:        PIL Image of the handwriting page / line. Passed through
                      to ``engine.run`` unchanged.
        filename:     Original filename of the uploaded image (e.g. "line_01.png").
                      Included in the ``meta`` event for frontend display.
        ground_truth: The human-verified transcription to compare against, or
                      ``None`` when the caller did not provide one. When
                      ``None``, ``cer`` and ``wer`` fields in result events
                      are ``null`` (JSON) / ``None`` (Python).

    Yields:
        NDJSON lines — each a ``json.dumps(event) + "\n"`` string. The
        sequence is always: meta, (result|error) × N, done.
    """
    # --- 1. Open the stream with a meta event ---------------------------------
    # The frontend uses ``has_ground_truth`` to decide whether to render the
    # CER/WER columns; when False those cells show "—" rather than 0.0%.
    yield _line(schemas.meta_event(filename, ground_truth is not None))

    # --- 2. Run each scenario in order ----------------------------------------
    for spec in _SPECS:
        try:
            # Time the full round-trip to the engine (including any network
            # latency for RunPod). perf_counter is monotonic and sub-millisecond
            # precision on all target platforms (Linux EC2, macOS dev machines).
            start = time.perf_counter()
            raw = engine.run(image, spec.prompt, spec.max_new_tokens)
            latency = round(time.perf_counter() - start, 3)

        except EngineError as exc:
            # Per-model error isolation: emit an error event for this scenario
            # and advance to the next. The stream stays alive.
            yield _line(schemas.error_event(spec.model, str(exc)))
            continue

        # --- 2a. Post-process the raw output depending on scenario ------------
        if spec.model == "m1":
            # M1 (baseline): the raw output *is* the transcription. Strip
            # leading/trailing whitespace that the model may include.
            text = raw.strip()
            log = M1_LOG
        else:
            # M2 (CoT): the raw output contains reasoning + 'Final:' answer.
            # parse_cot returns (final_answer, reasoning_log); the log goes into
            # the result event so the frontend can surface it in a tooltip.
            text, log = parse_cot(raw)

        # --- 2b. Compute quality metrics (only when ground truth is available) -
        if ground_truth is not None:
            # cer_metric / wer_metric return percentages (0–100 scale) via jiwer.
            # Round to 2 decimal places to match SP-1 evaluation table formatting.
            cer_value = round(cer_metric(ground_truth, text), 2)
            wer_value = round(wer_metric(ground_truth, text), 2)
        else:
            # No reference string supplied — use JSON null so the frontend can
            # distinguish "perfect 0 %" from "not measured".
            cer_value = wer_value = None

        # --- 2c. Emit the result event for this scenario ----------------------
        yield _line(schemas.result_event(
            model=spec.model,
            text=text,
            cer=cer_value,
            wer=wer_value,
            latency_seconds=latency,
            log=log,
            status_tag=spec.status_tag,
        ))

    # --- 3. Close the stream with a done event --------------------------------
    # Always emitted, even if all scenarios failed — the client must be able to
    # detect end-of-stream regardless of how many error events were emitted.
    yield _line(schemas.done_event())
