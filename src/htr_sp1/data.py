"""Dataset loading and example construction for IAM-line.

Two responsibilities:
1. `load_iam_splits` — fetch the official train/validation/test splits from the Hub.
2. `build_prompt` / `build_training_example` — turn a raw record into the inputs the
   PaliGemma processor needs (image + prompt prefix + label suffix).

The example-building functions take the processor as an argument (dependency injection) so
they are unit-testable with a fake and never trigger a model download.
"""
from __future__ import annotations

from typing import Any, Dict

from . import config


def load_iam_splits():
    """Load the official IAM-line splits from the Hub.

    Imported lazily so importing this module on a minimal laptop (no `datasets`) is cheap
    and our prompt/example unit tests stay fast.

    Returns:
        A DatasetDict with "train", "validation", and "test" splits. Each record has an
        "image" (PIL.Image) and a "text" (ground-truth transcription) field.
    """
    from datasets import load_dataset

    # Returns all splits; we keep them together so the caller picks what it needs.
    return load_dataset(config.DATASET_ID)


def build_prompt() -> str:
    """Return the fixed transcription prompt prefix (sourced from config)."""
    return config.TRANSCRIPTION_PROMPT


def build_training_example(record: Dict[str, Any], processor) -> Dict[str, Any]:
    """Encode one IAM record into model inputs WITH labels, for supervised fine-tuning.

    PaliGemma's processor builds the training labels for us when we pass the target text as
    `suffix`: it appends the suffix after the prompt and masks the prompt tokens in the loss.

    Args:
        record: An IAM record with "image" (PIL) and "text" (ground truth) keys.
        processor: A PaliGemmaProcessor (or compatible fake in tests).

    Returns:
        The processor's encoded batch (input_ids/attention_mask/pixel_values/labels...).
    """
    return processor(
        text=config.TRANSCRIPTION_PROMPT,  # the conditioning prompt prefix
        images=record["image"],            # the handwriting line image
        suffix=record["text"],             # the ground truth -> becomes the labels
        return_tensors="pt",
    )
