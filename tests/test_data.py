"""Tests for prompt construction and turning an IAM record into a training example."""
from htr_sp1 import config, data


def test_build_prompt_uses_config_constant():
    # The prompt must come from config so experiments tune it in one place.
    assert data.build_prompt() == config.TRANSCRIPTION_PROMPT


def test_build_training_example_passes_image_prompt_and_label(fake_processor):
    # A fake "PIL image" — our code should hand it straight to the processor.
    record = {"image": object(), "text": "the quick brown fox"}
    data.build_training_example(record, fake_processor)
    call = fake_processor.last_call
    assert call["images"] is record["image"]
    assert call["text"] == config.TRANSCRIPTION_PROMPT
    # The label is supplied as `suffix` so PaliGemma's processor builds the loss labels.
    assert call["suffix"] == "the quick brown fox"
    # PaliGemma must receive torch tensors ("pt"), not Python lists, or training fails.
    assert call["return_tensors"] == "pt"


def test_build_training_example_converts_grayscale_to_rgb(fake_processor):
    # IAM-line images are grayscale (mode "L" -> 2D). PaliGemma's processor needs 3-channel
    # RGB, else "Unsupported number of image dimensions: 2". The image must arrive as RGB.
    import pytest

    Image = pytest.importorskip("PIL.Image")
    gray = Image.new("L", (8, 4))  # grayscale line image, like IAM
    data.build_training_example({"image": gray, "text": "x"}, fake_processor)
    assert fake_processor.last_call["images"].mode == "RGB"
