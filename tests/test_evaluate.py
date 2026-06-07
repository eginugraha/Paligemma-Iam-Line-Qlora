"""Evaluation aggregates per-sample CER/WER into the headline test numbers. We inject a fake
transcriber so the test is deterministic and needs no model.
"""
from htr_sp1 import evaluate


def test_evaluate_split_averages_cer_and_wer():
    # Two records: first predicted perfectly, second has one wrong char ("fux").
    records = [
        {"image": object(), "text": "the quick brown fox"},
        {"image": object(), "text": "the quick brown fox"},
    ]
    # Fake transcriber: perfect on the first call, "fux" on the second.
    predictions = iter(["the quick brown fox", "the quick brown fux"])

    def fake_transcribe(image):
        return next(predictions)

    report = evaluate.evaluate_split(records, fake_transcribe)
    # Mean CER = (0 + 5.26) / 2 ≈ 2.63 ; mean WER = (0 + 25) / 2 = 12.5
    assert round(report["mean_cer"], 2) == 2.63
    assert round(report["mean_wer"], 2) == 12.5
    assert report["num_samples"] == 2
    assert len(report["per_sample"]) == 2
