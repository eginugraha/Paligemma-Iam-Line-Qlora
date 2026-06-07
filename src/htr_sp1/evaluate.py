"""Evaluate the fine-tuned model on a dataset split and produce the thesis baseline numbers.

`evaluate_split` takes a `transcribe` callable (so tests inject a fake and the notebook
injects the real `generate_transcription` bound to the model/processor). It returns mean
CER/WER plus per-sample rows for the appendix table.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

from . import metrics


def evaluate_split(records: Iterable[Dict[str, Any]], transcribe: Callable[[Any], str]) -> Dict[str, Any]:
    """Compute mean and per-sample CER/WER over a split.

    Args:
        records: Iterable of {"image", "text"} records (e.g. the IAM test split).
        transcribe: Function mapping an image to a predicted string. In Colab this is
            `lambda img: generate_transcription(model, processor, img)`; in tests it's a fake.

    Returns:
        Dict with keys: mean_cer, mean_wer, num_samples, per_sample (list of row dicts).
    """
    per_sample = []
    for record in records:
        prediction = transcribe(record["image"])
        ground_truth = record["text"]
        # Per-sample errors feed both the average and the appendix table for Bab 4.
        per_sample.append(
            {
                "ground_truth": ground_truth,
                "prediction": prediction,
                "cer": metrics.cer(ground_truth, prediction),
                "wer": metrics.wer(ground_truth, prediction),
            }
        )

    num = len(per_sample)
    if num == 0:  # guard against an empty split so we never divide by zero.
        return {"mean_cer": 0.0, "mean_wer": 0.0, "num_samples": 0, "per_sample": []}

    return {
        "mean_cer": sum(row["cer"] for row in per_sample) / num,
        "mean_wer": sum(row["wer"] for row in per_sample) / num,
        "num_samples": num,
        "per_sample": per_sample,
    }
