"""RunPodEngine is an HTTP client to a RunPod Serverless /runsync endpoint. We mock HTTP
with pytest-httpx so no GPU/network is needed: assert the request shape and that responses
and failures map correctly."""
import httpx
import pytest
from PIL import Image

from htr_sp2 import engine
from htr_sp2.engines.runpod import RunPodEngine


def _engine():
    return RunPodEngine(endpoint_id="ep123", api_key="key", timeout=5.0)


def _image():
    return Image.new("RGB", (4, 4), (255, 255, 255))


def test_run_posts_payload_and_returns_text(httpx_mock):
    httpx_mock.add_response(json={"output": {"text": "the quick brown fox"}})
    out = _engine().run(_image(), prompt="transcribe\n", max_new_tokens=64)
    assert out == "the quick brown fox"

    request = httpx_mock.get_request()
    assert request.url == "https://api.runpod.ai/v2/ep123/runsync"
    assert request.headers["authorization"] == "Bearer key"
    body = httpx.Request("POST", request.url, content=request.content).read()
    import json
    sent = json.loads(body)
    assert sent["input"]["prompt"] == "transcribe\n"
    assert sent["input"]["max_new_tokens"] == 64
    assert sent["input"]["image_b64"]  # non-empty base64


def test_run_raises_engine_error_on_http_error(httpx_mock):
    httpx_mock.add_response(status_code=500)
    with pytest.raises(engine.EngineError):
        _engine().run(_image(), prompt="p", max_new_tokens=64)


def test_run_raises_engine_error_on_timeout(httpx_mock):
    httpx_mock.add_exception(httpx.TimeoutException("too slow"))
    with pytest.raises(engine.EngineError):
        _engine().run(_image(), prompt="p", max_new_tokens=64)


def test_run_raises_engine_error_on_bad_response_shape(httpx_mock):
    httpx_mock.add_response(json={"unexpected": True})
    with pytest.raises(engine.EngineError):
        _engine().run(_image(), prompt="p", max_new_tokens=64)
