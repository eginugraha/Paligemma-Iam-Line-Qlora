"""Tests for SP-5 schemas: fold_results and eval_rows_from_results.

These tests cover the pure fold logic that collapses raw NDJSON stream events
into the persisted results dict and then expands that dict into typed row objects.
No DB or HTTP fixtures are needed — all inputs are plain Python dicts/lists.
"""
from htr_sp5.schemas import fold_results, EvalResultRow, eval_rows_from_results


# ---------------------------------------------------------------------------
# Shared fixture — a minimal but representative event sequence
# ---------------------------------------------------------------------------

def _events():
    """Return a sample NDJSON event sequence with two result events, one error, one meta, one done.

    Mirrors a real /v1/detect stream:
      - meta:   file announcement (not a result — should be discarded)
      - result: m1 succeeded with metrics
      - error:  m2 failed (should be silently dropped from the output dict)
      - result: m3 succeeded with corrected output
      - done:   terminal sentinel (should be discarded)
    """
    return [
        {"event": "meta", "filename": "a.png", "has_ground_truth": True},
        {"event": "result", "model": "m1", "text": "the cat", "cer": 5.0, "wer": 10.0,
         "latency_seconds": 0.7, "log": "Direct.", "status_tag": "Raw Output"},
        {"event": "error", "model": "m2", "message": "boom"},
        {"event": "result", "model": "m3", "text": "the cat", "cer": 0.0, "wer": 0.0,
         "latency_seconds": 1.1, "log": "RAG: ...", "status_tag": "Corrected"},
        {"event": "done"},
    ]


# ---------------------------------------------------------------------------
# Tests for fold_results
# ---------------------------------------------------------------------------

def test_fold_results_keeps_only_result_events_keyed_by_model():
    """fold_results must keep exactly the two successful result events and nothing else.

    Assertions:
    - Only models with ``event == "result"`` appear as keys.
    - Payload values are preserved faithfully (text, cer, status_tag sampled).
    - The ``"event"`` discriminator and ``"model"`` key are stripped from each value dict
      (they become the dict key and are redundant noise for downstream consumers).
    """
    out = fold_results(_events())
    assert set(out.keys()) == {"m1", "m3"}            # error + meta + done dropped
    assert out["m1"]["text"] == "the cat"
    assert out["m1"]["cer"] == 5.0
    assert out["m3"]["status_tag"] == "Corrected"
    assert "event" not in out["m1"] and "model" not in out["m1"]


def test_fold_results_skips_result_event_with_missing_model_key():
    """fold_results must silently skip a ``result`` event that has no ``model`` key.

    A result event without a model is unusable (no dict key to store it under) so
    the implementation uses ``evt.get("model")`` and ``continue``-s when it is None.
    Other well-formed events in the same stream must still be processed normally.
    """
    events = [
        {"event": "result", "text": "oops", "cer": 1.0, "wer": 2.0,
         "latency_seconds": 0.1, "log": "x", "status_tag": "Raw Output"},  # no "model"
        {"event": "result", "model": "m1", "text": "hi", "cer": 0.0, "wer": 0.0,
         "latency_seconds": 0.5, "log": "OK", "status_tag": "Raw Output"},
    ]
    out = fold_results(events)
    assert set(out.keys()) == {"m1"}     # malformed event skipped; m1 present
    assert out["m1"]["text"] == "hi"


def test_fold_results_missing_optional_fields_become_none():
    """A ``result`` event that omits cer and wer must yield those keys as None.

    This is the no-ground-truth case: the orchestrator omits ``cer`` and ``wer``
    when the user did not supply a reference transcription.  The fold must still
    produce a well-formed payload dict with None for those two fields.
    """
    events = [{"event": "result", "model": "m1", "text": "hi", "latency_seconds": 0.3,
               "log": "x", "status_tag": "Raw Output"}]  # no cer / wer
    out = fold_results(events)
    assert out["m1"]["cer"] is None and out["m1"]["wer"] is None
    assert out["m1"]["text"] == "hi"


# ---------------------------------------------------------------------------
# Tests for eval_rows_from_results
# ---------------------------------------------------------------------------

def test_eval_rows_from_results_emits_one_row_per_scenario():
    """eval_rows_from_results must produce one EvalResultRow for each key in the results dict.

    Assertions:
    - Exactly the scenarios that survived fold_results appear in the output list.
    - Each row carries the correct sample_id and ground_truth threaded from the caller.
    - All metric/text fields are passed through unchanged from the folded dict.
    - The row is an instance of EvalResultRow (frozen dataclass, not a plain dict).
    """
    results = fold_results(_events())
    rows = eval_rows_from_results("sample-42", "the cat", results)
    assert {r.scenario for r in rows} == {"m1", "m3"}
    r1 = next(r for r in rows if r.scenario == "m1")
    assert isinstance(r1, EvalResultRow)
    assert r1.sample_id == "sample-42"
    assert r1.ground_truth == "the cat"
    assert r1.text == "the cat" and r1.cer == 5.0 and r1.latency_seconds == 0.7


def test_eval_rows_from_results_threads_none_ground_truth():
    """eval_rows_from_results must thread ground_truth=None into every row unchanged.

    When the user uploads an image without a reference transcription, ground_truth
    is None.  Every EvalResultRow produced for that sample must carry None in the
    ground_truth field — it must not be coerced to an empty string or any other value.
    """
    results = {"m1": {"text": "hi", "cer": None, "wer": None,
                      "latency_seconds": 0.3, "log": "x", "status_tag": "Raw Output"}}
    rows = eval_rows_from_results("s1", None, results)
    assert all(r.ground_truth is None for r in rows)
