"""Character and Word Error Rate, expressed as percentages (0–100).

Both are edit-distance (Levenshtein) based: the minimum number of insertions, deletions,
and substitutions to turn the prediction into the reference, divided by the reference
length. We delegate the math to `jiwer` (well-tested) and only convert to percent here so
our numbers line up with the PRD's example output (e.g. CER 5.26, WER 25.0).
"""
from __future__ import annotations

import jiwer


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate as a percentage.

    Args:
        reference: The ground-truth transcription.
        hypothesis: The model's predicted transcription.

    Returns:
        Edit distance over characters / reference character count, times 100.
    """
    # jiwer.cer returns a fraction in [0, ~]; multiply by 100 for a human-readable percent.
    return jiwer.cer(reference, hypothesis) * 100.0


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate as a percentage.

    Args:
        reference: The ground-truth transcription.
        hypothesis: The model's predicted transcription.

    Returns:
        Edit distance over words / reference word count, times 100.
    """
    return jiwer.wer(reference, hypothesis) * 100.0
