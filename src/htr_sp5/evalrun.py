"""Batch-evaluation core: run M1-M4 over samples and persist eval rows.

Kept separate from the CLI so it can be tested with a FakeEngine and a recording store — no GPU,
no dataset download, no database. The CLI (scripts/eval_sp5.py) supplies the real objects.

Architecture note (SP-5, Chapter 4 — Batch Evaluation)
--------------------------------------------------------
The function ``run_eval`` is the *engine-agnostic kernel* of the SP-5 pipeline.  It receives
three injected collaborators:

  - **engine**    — any object satisfying ``InferenceEngine.run(image, prompt, max_new_tokens)``
                    (the Protocol from htr_sp2.engine).  In production this is a ``RunPodEngine``
                    or ``LocalEngine``; in tests it is a ``FakeEngine`` that returns a fixed string
                    without touching any model weights.

  - **corrector** — either a live ``RagCorrector`` from htr_sp3 (enables M3/M4 RAG scenarios) or
                    ``None`` (skips M3/M4 entirely, yielding only M1 and M2 per sample).

  - **store**     — any object satisfying the two-method persistence interface:
                    ``create_eval_run(...)`` → ``int`` run_id, and
                    ``insert_eval_results(run_id, rows)``.
                    In production this is ``Sp5Store`` (store.py, Task 3); in tests it is a
                    ``RecordingStore`` that just appends to a list.

This dependency-injection design means the whole evaluation pipeline is unit-testable in under a
second on any laptop: no GPU, no Docker, no Postgres connection required.

Data flow per sample
--------------------
1. Call ``detect_stream(engine, image, sample_id, ground_truth, corrector=corrector)`` from
   htr_sp2.orchestrator.  This is a generator that yields NDJSON *strings*, one per event.

2. ``json.loads`` each string back into a Python dict (the orchestrator serialised them for HTTP
   streaming; we de-serialise to use the pure-Python ``fold_results`` helper).

3. ``fold_results(events)`` reduces the list of event dicts to a compact
   ``{scenario: {text, cer, wer, latency_seconds, log, status_tag}}`` mapping, discarding
   the ``meta``, ``error``, and ``done`` bookkeeping events.

4. ``eval_rows_from_results(sample_id, ground_truth, results)`` expands that mapping into one
   ``EvalResultRow`` frozen dataclass per successful scenario.

5. ``store.insert_eval_results(run_id, rows)`` persists the rows for the sample.

The store's ``create_eval_run`` is called **once** before the sample loop so all sample rows
share a single ``run_id`` foreign key — a natural grouping for the dashboard's "evaluation run"
concept.
"""
from __future__ import annotations

import json
from typing import Iterable

# detect_stream is the SP-2 orchestrator entrypoint.  It accepts a duck-typed engine and
# corrector, drives M1/M2 (and optionally M3/M4), and yields NDJSON strings.  Importing it
# here creates the only tight coupling between SP-5 and SP-2 — everything else (engine,
# corrector, store) is injected by the caller.
from htr_sp2.orchestrator import detect_stream

# fold_results and eval_rows_from_results live in htr_sp5.schemas (Task 2).
# fold_results  → reduces raw event dicts into a compact per-scenario dict.
# eval_rows_from_results → expands that dict into EvalResultRow frozen dataclasses.
# Both are pure functions (no I/O) so the same path is exercised identically in tests and
# production — changing the serialisation format in orchestrator.py would break both at once,
# which is the correct failure mode.
from htr_sp5.schemas import eval_rows_from_results, fold_results


def run_eval(
    samples: Iterable[dict],
    engine,
    corrector,
    store,
    *,
    dataset: str,
    model_ref: str | None,
) -> int:
    """Evaluate each sample through detect_stream and persist one eval_run + N*scenario rows.

    This is the single public API of evalrun.py.  The CLI calls it with real objects; tests
    call it with fakes.  Either way the logic is identical — only the collaborators differ.

    Sample format
    -------------
    Each element in ``samples`` must be a dict with three keys:
      - ``"sample_id"``    : ``str``  — stable identifier for this image (e.g. IAM line id or
                             a zero-padded index).  Used as the primary key in eval_result rows
                             and for joining back to the uploaded image in the dashboard.
      - ``"image"``        : ``PIL.Image.Image`` — the handwriting image in RGB mode.  Passed
                             through to engine.run unchanged.
      - ``"ground_truth"`` : ``str | None`` — the reference transcription.  When provided,
                             CER/WER are computed by the orchestrator; when None those fields
                             are None in the stored rows.

    Scenarios persisted
    -------------------
    Without a corrector: M1 (baseline direct) + M2 (CoT) → 2 rows per sample.
    With a corrector:    M1 + M2 + M3 (RAG on M1) + M4 (RAG on M2) → up to 4 rows per sample.
    Failed scenarios produce an ``error`` event in the stream and are silently omitted from
    the stored rows (``fold_results`` only keeps ``result`` events).

    Args:
        samples:   An iterable of sample dicts (see above).  Materialised to a list
                   immediately so ``len`` is available for ``create_eval_run``.
        engine:    Duck-typed InferenceEngine with ``run(image, prompt, max_new_tokens) -> str``.
        corrector: Duck-typed RagCorrector (htr_sp3) with ``correct(text) -> (str, list)``, or
                   ``None`` to skip M3/M4.
        store:     Persistence backend with ``create_eval_run`` and ``insert_eval_results``.
        dataset:   Human-readable dataset label stored on the run (e.g. ``"iam-line-test"``).
        model_ref: Optional HuggingFace repo id or git tag for the model/adapter used, so the
                   dashboard can group runs by model version.  ``None`` is fine for quick smoke
                   tests where the version is unknown.

    Returns:
        The integer ``run_id`` returned by ``store.create_eval_run``; callers can use it to
        query the dashboard or pass to downstream steps.
    """
    # Materialise the iterable once so we can pass n_samples to create_eval_run without
    # consuming the iterator — and so we iterate it only once below.
    samples = list(samples)

    # --- 1. Open an eval_run record in the store ---------------------------------
    # create_eval_run records the high-level metadata for this batch run (which dataset,
    # how many samples, whether RAG was enabled).  All per-sample rows reference this
    # run_id as a foreign key, which lets the dashboard aggregate metrics per run.
    run_id = store.create_eval_run(
        dataset=dataset,
        n_samples=len(samples),
        model_ref=model_ref,
        rag_enabled=corrector is not None,
    )

    # --- 2. Evaluate each sample in order -----------------------------------------
    for s in samples:
        # detect_stream yields NDJSON *strings* (each is ``json.dumps(event) + "\n"``).
        # We parse them back to dicts here because fold_results and eval_rows_from_results
        # work on plain Python dicts — they have no knowledge of the wire format.
        # This round-trip (serialize → deserialize) is intentional: it exercises the exact
        # same code path used by the HTTP streaming endpoint (SP-2 API), so any JSON
        # serialisation bug surfaces in both contexts simultaneously.
        raw_lines = detect_stream(
            engine,
            s["image"],
            s["sample_id"],          # used as filename in the meta event; also the row key
            s.get("ground_truth"),   # None is valid — .get() avoids KeyError on optional field
            corrector=corrector,
        )
        events = [json.loads(line) for line in raw_lines]

        # fold_results reduces [meta, result, result, ..., done] to {model: {text, cer, ...}}.
        # Scenarios that emitted an ``error`` event are absent from the returned dict — no row
        # is persisted for them, which is the correct behaviour (missing data > corrupt data).
        results = fold_results(events)

        # eval_rows_from_results lifts the flat dict into a list of typed EvalResultRow objects.
        # ground_truth is threaded in explicitly because it is not present in the stream events
        # (the orchestrator receives it as a parameter for metric computation but does not echo
        # it back in the result event — it would be redundant and add payload overhead).
        rows = eval_rows_from_results(s["sample_id"], s.get("ground_truth"), results)

        # Persist all rows for this sample in a single call.  The store is responsible for
        # batching into one INSERT; we hand it the full list so it can do that efficiently.
        store.insert_eval_results(run_id, rows)

    return run_id
