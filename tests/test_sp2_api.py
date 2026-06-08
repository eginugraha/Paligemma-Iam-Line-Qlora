"""End-to-end through FastAPI's TestClient using the default fake engine. We post a real
(tiny) PNG so the image-decode path runs, then parse the streamed NDJSON lines."""
import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from htr_sp2.api import app

client = TestClient(app)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_detect_streams_meta_results_done():
    resp = client.post(
        "/v1/detect",
        files={"file": ("line_01.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = _parse_ndjson(resp.text)
    assert [e["event"] for e in events] == ["meta", "result", "result", "done"]
    assert events[0]["filename"] == "line_01.png"
    assert events[0]["has_ground_truth"] is False


def test_detect_with_ground_truth_sets_flag_and_metrics():
    resp = client.post(
        "/v1/detect",
        files={"file": ("line_01.png", _png_bytes(), "image/png")},
        data={"ground_truth": "the quick brown fox"},
    )
    events = _parse_ndjson(resp.text)
    assert events[0]["has_ground_truth"] is True
    # Default FakeEngine returns "the quick brown fox" -> perfect match -> 0.0
    assert events[1]["cer"] == 0.0 and events[1]["wer"] == 0.0


def test_detect_rejects_undecodable_image():
    resp = client.post(
        "/v1/detect",
        files={"file": ("bad.png", b"not really an image", "image/png")},
    )
    assert resp.status_code == 422
