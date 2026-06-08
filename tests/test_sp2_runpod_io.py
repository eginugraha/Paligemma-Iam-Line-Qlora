"""runpod_io is the wire format shared by the client (RunPodEngine) and the server
(handler). Round-tripping an image through encode->decode and payload->parse keeps both
sides in sync and is fully testable on CPU."""
import io

from PIL import Image

from htr_sp2 import runpod_io


def _tiny_image():
    return Image.new("RGB", (4, 4), color=(255, 255, 255))


def test_encode_then_decode_roundtrips_size():
    b64 = runpod_io.encode_image(_tiny_image())
    assert isinstance(b64, str) and b64  # non-empty base64 string
    img = runpod_io.decode_image(b64)
    assert img.size == (4, 4)


def test_build_payload_shape():
    payload = runpod_io.build_payload(_tiny_image(), prompt="p", max_new_tokens=7)
    assert set(payload["input"].keys()) == {"image_b64", "prompt", "max_new_tokens"}
    assert payload["input"]["prompt"] == "p"
    assert payload["input"]["max_new_tokens"] == 7


def test_parse_input_returns_image_prompt_tokens():
    payload = runpod_io.build_payload(_tiny_image(), prompt="p", max_new_tokens=7)
    parsed = runpod_io.parse_input(payload)  # server reads {"input": {...}}
    assert parsed["prompt"] == "p"
    assert parsed["max_new_tokens"] == 7
    assert parsed["image"].size == (4, 4)


def test_parse_output_extracts_text():
    assert runpod_io.parse_output({"output": {"text": "hello"}}) == "hello"


def test_parse_output_raises_on_bad_shape():
    import pytest
    with pytest.raises(KeyError):
        runpod_io.parse_output({"unexpected": True})
