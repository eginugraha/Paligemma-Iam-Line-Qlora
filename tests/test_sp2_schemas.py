"""Each streamed line is one event dict. These builders are the single source of truth for
the NDJSON contract the frontend (SP-4) and batch eval (SP-5) consume."""
from htr_sp2 import schemas


def test_meta_event():
    assert schemas.meta_event("line_01.png", True) == {
        "event": "meta", "filename": "line_01.png", "has_ground_truth": True,
    }


def test_result_event():
    ev = schemas.result_event(
        model="m1", text="the quick brown fox", cer=5.26, wer=25.0,
        latency_seconds=0.78, log="done.", status_tag="Raw Output",
    )
    assert ev == {
        "event": "result", "model": "m1", "text": "the quick brown fox",
        "cer": 5.26, "wer": 25.0, "latency_seconds": 0.78,
        "log": "done.", "status_tag": "Raw Output",
    }


def test_result_event_allows_null_metrics():
    ev = schemas.result_event(
        model="m1", text="x", cer=None, wer=None,
        latency_seconds=0.1, log="done.", status_tag="Raw Output",
    )
    assert ev["cer"] is None and ev["wer"] is None


def test_error_and_done_events():
    assert schemas.error_event("m2", "boom") == {"event": "error", "model": "m2", "message": "boom"}
    assert schemas.done_event() == {"event": "done"}
