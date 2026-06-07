"""The public inference interface for the fine-tuned model.

THIS IS THE CONTRACT SP-2 CONSUMES. Keep it tiny and stable: give it a loaded model, a
processor, and a PIL image, and it returns the predicted transcription string. M1 (baseline)
in SP-2 is literally this call; M2 (CoT) will reuse the same model with a different prompt.
"""
from __future__ import annotations

from . import config


def generate_transcription(model, processor, image, max_new_tokens: int = config.MAX_TARGET_TOKENS) -> str:
    """Transcribe a single handwriting-line image.

    Args:
        model: A loaded PaliGemma model (fine-tuned, or base+adapter, or a test fake).
        processor: The matching PaliGemmaProcessor (or a test fake).
        image: A PIL.Image of one handwriting line.
        max_new_tokens: Generation cap; IAM lines are short so the default is small.

    Returns:
        The predicted transcription, whitespace-stripped.
    """
    # Encode WITHOUT a suffix — at inference we have no ground truth; the model generates it.
    inputs = processor(
        text=config.TRANSCRIPTION_PROMPT,
        images=image,
        suffix=None,
        return_tensors="pt",
    )
    # `.to(...)` is a no-op on fakes; on a real run it moves tensors to the model's device.
    inputs = inputs.to(getattr(model, "device", "cpu"))

    # Greedy decode (do_sample defaults False) for reproducible, deterministic transcriptions.
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode the first (only) sequence and strip padding/whitespace for clean metrics.
    text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return text.strip()
