"""detect_stream is the heart of the backend: run M1 then M2 against an engine, compute
CER/WER when ground truth is given, and yield NDJSON lines. We drive it with FakeEngine so
it is deterministic. Latency is timing-based, so we assert its type/sign, not a value."""
import json

from htr_sp2 import config
from htr_sp2.engines.fake import FakeEngine
from htr_sp2.orchestrator import detect_stream


def _events(lines):
    # Each yielded line is a JSON object + newline; parse them back for assertions.
    return [json.loads(line) for line in lines]


def test_stream_emits_meta_two_results_done_in_order():
    eng = FakeEngine(responses=["the quick brown fox", "Reasoning: r\nFinal: the quick brown fox"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert [e["event"] for e in events] == ["meta", "result", "result", "done"]
    assert events[0] == {"event": "meta", "filename": "x.png", "has_ground_truth": False}
    assert events[1]["model"] == "m1" and events[2]["model"] == "m2"


def test_m1_uses_raw_text_and_tag():
    eng = FakeEngine(responses=["the quick brown fox", "Final: ignored"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    m1 = events[1]
    assert m1["text"] == "the quick brown fox"
    assert m1["status_tag"] == config.M1_STATUS_TAG
    assert m1["log"] == "Direct visual token translation completed."


def test_m2_is_parsed_into_text_and_reasoning_log():
    eng = FakeEngine(responses=["x", "Reasoning: loop on d\nFinal: medical"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    m2 = events[2]
    assert m2["text"] == "medical"
    assert m2["log"] == "Reasoning: loop on d"
    assert m2["status_tag"] == config.M2_STATUS_TAG


def test_metrics_null_without_ground_truth():
    eng = FakeEngine(responses=["abc", "Final: abc"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert events[1]["cer"] is None and events[1]["wer"] is None


def test_metrics_computed_with_ground_truth():
    # Perfect match -> 0.0 CER/WER; reuses htr_sp1.metrics (jiwer).
    eng = FakeEngine(responses=["the quick brown fox", "Final: the quick brown fox"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth="the quick brown fox"))
    assert events[1]["cer"] == 0.0 and events[1]["wer"] == 0.0


def test_latency_is_non_negative_float():
    eng = FakeEngine(responses=["a", "Final: a"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert isinstance(events[1]["latency_seconds"], float) and events[1]["latency_seconds"] >= 0.0


def test_engine_error_on_one_model_emits_error_and_continues():
    # M1 (call 0) fails; M2 (call 1) still runs. Stream stays alive.
    eng = FakeEngine(responses=["x", "Final: medical"], fail_on={0})
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert [e["event"] for e in events] == ["meta", "error", "result", "done"]
    assert events[1] == {"event": "error", "model": "m1", "message": "fake failure on call 0"}
    assert events[2]["model"] == "m2" and events[2]["text"] == "medical"


def test_engine_called_with_correct_prompts_and_caps():
    eng = FakeEngine(responses=["a", "Final: b"])
    list(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert eng.calls[0]["prompt"] == config.M1_PROMPT
    assert eng.calls[0]["max_new_tokens"] == config.M1_MAX_NEW_TOKENS
    assert eng.calls[1]["prompt"] == config.M2_PROMPT
    assert eng.calls[1]["max_new_tokens"] == config.M2_MAX_NEW_TOKENS
