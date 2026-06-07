"""`generate_transcription` is the public interface SP-2 will import. We test that it:
  - encodes the image + prompt,
  - calls the model's generate,
  - decodes and strips whitespace.
Using fakes keeps it laptop-fast and deterministic.
"""
from htr_sp1 import inference


def test_generate_transcription_returns_clean_text(fake_model, fake_processor):
    fake_processor.next_decoded = "  the quick brown fox  "  # decode() will return this
    out = inference.generate_transcription(fake_model, fake_processor, image=object())
    # Output is stripped so downstream CER/WER aren't polluted by padding whitespace.
    assert out == "the quick brown fox"


def test_generate_transcription_feeds_prompt_and_image(fake_model, fake_processor):
    img = object()
    inference.generate_transcription(fake_model, fake_processor, image=img)
    from htr_sp1 import config
    assert fake_processor.last_call["images"] is img
    assert fake_processor.last_call["text"] == config.TRANSCRIPTION_PROMPT
    # At inference there is NO suffix (we have no label; the model must produce the text).
    assert fake_processor.last_call["suffix"] is None
