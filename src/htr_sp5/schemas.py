"""Plain dataclasses for SP-5 rows + pure helpers that fold NDJSON stream events into them.

Keeping this logic free of any DB/HTTP dependency makes the persistence path fully unit-testable:
the same ``fold_results`` output feeds both the upload history (JSONB blob) and the batch-eval rows.

NDJSON → dict → rows pipeline
-------------------------------
The SP-2 orchestrator (``src/htr_sp2/orchestrator.py``) streams one JSON object per line
to the caller.  Each object carries a mandatory ``"event"`` discriminator:

    meta      — one per request; announces filename and ground-truth availability
    result    — one per scenario that completed successfully; carries text + metrics
    error     — one per scenario that raised an exception; only a message is included
    done      — exactly one final line; signals the stream is fully consumed

SP-5 batch evaluation reads these events and needs to:

  1. **Fold** them into a compact ``{model: {text, cer, wer, latency_seconds, log, status_tag}}``
     dict that can be stored verbatim as a Postgres JSONB blob in ``upload_result.results``.

  2. **Expand** that dict into one ``EvalResultRow`` per scenario so the persistence layer
     (``store.py``, Task 3) can do a bulk INSERT into the ``eval_result`` table.

This module handles both steps.  It intentionally has **no** DB or HTTP imports so that any
consumer can be tested with plain Python lists — no running database required.

Why dataclasses, not plain dicts?
-----------------------------------
``EvalResultRow`` uses a frozen dataclass (``frozen=True``) rather than a plain dict for two
reasons:

  * **Type safety** — callers and tests get IDE auto-complete and static type checking on
    field names, catching typos that would otherwise silently store ``None`` in Postgres.

  * **Immutability** — ``frozen=True`` makes each row hashable and prevents accidental
    in-place mutation after construction; the persistence layer should never modify a row
    object, only read it.

Field names in ``EvalResultRow`` match the ``eval_result`` database column names (set in
``store.py``) exactly, so the persistence layer can use ``dataclasses.asdict`` without
any key remapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# The six payload fields we persist per scenario.  These are the fields from a ``result``
# event after stripping the two discriminator keys (``"event"`` and ``"model"``).
# Listing them explicitly here is the single source of truth for what a persisted result
# contains: if the SP-2 event shape gains a new field in the future, add it here, in
# ``EvalResultRow``, and in the ``eval_result`` table migration — nowhere else.
_RESULT_FIELDS = ("text", "cer", "wer", "latency_seconds", "log", "status_tag")


# ---------------------------------------------------------------------------
# Dataclass — one row destined for the eval_result table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResultRow:
    """One (sample × scenario) row destined for the ``eval_result`` table.

    Instances are produced by ``eval_rows_from_results`` and consumed by ``store.py``.
    Field names match database column names so ``dataclasses.asdict(row)`` can be passed
    directly to a SQLAlchemy ``INSERT`` statement without key remapping.

    Attributes:
        sample_id:        Foreign key into ``upload_result`` (identifies which image this
                          measurement belongs to).  Typically a UUID string or a short
                          human-readable slug supplied by the batch evaluator.
        scenario:         The scenario identifier — ``"m1"`` (baseline), ``"m2"`` (CoT),
                          ``"m3"`` (RAG), or ``"m4"`` (CoT+RAG).  Taken from the ``model``
                          field of the original ``result`` event.
        text:             The transcription produced by the HTR model for this scenario.
                          ``None`` if the event was missing the field (should not happen in
                          a well-formed stream, but we guard with ``.get``).
        ground_truth:     The reference transcription supplied by the user (may be ``None``
                          when the image was uploaded without a ground-truth string).
        cer:              Character Error Rate as a percentage (0–100).  ``None`` when no
                          ground truth was available so the metric could not be computed.
        wer:              Word Error Rate as a percentage (0–100).  Same nullability rule
                          as ``cer``.
        latency_seconds:  Wall-clock seconds from when the scenario started until the model
                          returned its result.  Used in the thesis performance analysis.
        log:              Short human-readable status string from the orchestrator
                          (e.g. ``"Inference OK"`` or ``"RAG: 3 candidates retrieved"``).
        status_tag:       One of the fixed UI badge strings (e.g. ``"Raw Output"``,
                          ``"CoT Output"``, ``"Corrected"``).  The dashboard maps these to
                          badge colours without further parsing.
    """

    # ---------- identity ----------
    sample_id: str
    """Foreign key identifying the source image / upload_result row."""

    scenario: str
    """Scenario slug: 'm1' | 'm2' | 'm3' | 'm4'."""

    # ---------- transcription ----------
    text: str | None
    """HTR output text for this scenario."""

    ground_truth: str | None
    """Reference transcription; None when the user did not supply one."""

    # ---------- metrics ----------
    cer: float | None
    """Character Error Rate %; None when ground_truth is absent."""

    wer: float | None
    """Word Error Rate %; None when ground_truth is absent."""

    latency_seconds: float | None
    """Wall-clock inference time in seconds."""

    # ---------- diagnostics ----------
    log: str | None
    """Short human-readable status string from the orchestrator."""

    status_tag: str | None
    """UI badge label (e.g. 'Raw Output', 'CoT Output', 'Corrected')."""


# ---------------------------------------------------------------------------
# Pure fold helpers
# ---------------------------------------------------------------------------

def fold_results(events: Iterable[dict]) -> dict:
    """Reduce a sequence of NDJSON events to ``{model: {text, cer, wer, latency_seconds, log, status_tag}}``.

    Only ``result`` events are kept; ``meta`` / ``error`` / ``done`` are silently ignored.
    Errored scenarios simply do not appear in the output — the dashboard and history views
    render whatever scenarios succeeded and leave empty cells for those that did not.

    The returned dict is intentionally plain (not typed) because it is stored verbatim as a
    JSONB blob via ``json.dumps``.  Using a typed object here would require an extra
    serialisation step before storage.

    Args:
        events: An iterable of dicts, each representing one line of an NDJSON stream.
                Extra keys beyond the known ``_RESULT_FIELDS`` are silently dropped so
                that future additions to the event format don't break old readers.

    Returns:
        A dict ``{model_name: payload_dict}`` where ``payload_dict`` contains exactly the
        six fields in ``_RESULT_FIELDS`` (values may be ``None`` if missing from the event).

    Example::

        >>> events = [
        ...     {"event": "meta", "filename": "a.png", "has_ground_truth": True},
        ...     {"event": "result", "model": "m1", "text": "hi", "cer": 0.0,
        ...      "wer": 0.0, "latency_seconds": 0.5, "log": "OK", "status_tag": "Raw Output"},
        ...     {"event": "done"},
        ... ]
        >>> fold_results(events)
        {'m1': {'text': 'hi', 'cer': 0.0, 'wer': 0.0, 'latency_seconds': 0.5, 'log': 'OK', 'status_tag': 'Raw Output'}}
    """
    out: dict[str, dict] = {}
    for evt in events:
        # Skip every event type except ``result`` — meta, error, and done carry no
        # per-scenario metrics and must not pollute the results dict.
        if evt.get("event") == "result":
            # A result event without a ``model`` key is unusable: we would have no key
            # under which to store the payload, and silently assigning it to ``None``
            # would corrupt the dict.  Skip it rather than hard-raising KeyError so
            # that a single malformed line in the stream does not abort the whole fold.
            model = evt.get("model")
            if model is None:
                continue

            # Build a clean payload dict using only the known result fields.
            # Using ``evt.get(k)`` rather than ``evt[k]`` means a stream that omits an
            # optional field (e.g. cer/wer when ground truth is absent) yields ``None``
            # rather than raising KeyError.
            out[model] = {k: evt.get(k) for k in _RESULT_FIELDS}
    return out


def eval_rows_from_results(
    sample_id: str,
    ground_truth: str | None,
    results: dict,
) -> list[EvalResultRow]:
    """Expand a folded results dict into one ``EvalResultRow`` per scenario.

    This is the bridge between the JSONB blob stored in ``upload_result.results``
    and the normalised ``eval_result`` table rows.  It is called by the batch
    evaluator after ``fold_results`` has already stripped the noise events.

    Args:
        sample_id:    Identifier for the source image, threaded into every row so the
                      persistence layer can join back to ``upload_result``.
        ground_truth: The reference transcription for this image (may be ``None``).
                      Threaded into every row; not present in the raw ``result`` event
                      because it was supplied by the user outside the inference pipeline.
        results:      The dict returned by ``fold_results`` —
                      ``{model: {text, cer, wer, latency_seconds, log, status_tag}}``.

    Returns:
        A list of ``EvalResultRow`` objects, one per key in ``results``.
        The order is not guaranteed (dict iteration order in CPython ≥ 3.7 is insertion
        order, but callers should not rely on it for correctness).

    Example::

        >>> results = {'m1': {'text': 'hi', 'cer': 0.0, 'wer': 0.0,
        ...                   'latency_seconds': 0.5, 'log': 'OK', 'status_tag': 'Raw Output'}}
        >>> rows = eval_rows_from_results("img-001", "hi", results)
        >>> rows[0].scenario
        'm1'
        >>> rows[0].ground_truth
        'hi'
    """
    rows: list[EvalResultRow] = []
    for scenario, r in results.items():
        # Construct one frozen row per scenario.  ground_truth is threaded in from the
        # caller because it lives on the upload_result row, not inside the result event.
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
