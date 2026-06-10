"""tune_threshold scans candidate thresholds and returns the one with the lowest mean CER on the
provided (prediction, ground_truth) pairs, plus the per-threshold curve. It is store/model
agnostic: we inject an in-memory corrector factory so no DB or model is needed.
"""
from htr_sp3 import tune, vectorize
from htr_sp3.corrector import RagCorrector
from htr_sp3.store import InMemoryVectorStore

VOCAB = ["medical", "record", "the", "patient", "was"]


def _make_corrector(threshold):
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in VOCAB])
    return RagCorrector(store=store, vocab=set(VOCAB), threshold=threshold)


def test_tune_picks_threshold_that_minimizes_cer():
    # Predictions have OCR errors that correction fixes; ground truth is the clean text.
    pairs = [("medisal", "medical"), ("recyrd", "record"), ("the", "the")]
    result = tune.tune_threshold(pairs, _make_corrector, thresholds=[0.0, 0.34])

    # 0.0 corrects nothing (high CER); 0.34 fixes the typos (CER 0) -> best is 0.34.
    assert result["best_threshold"] == 0.34
    assert result["best_cer"] < result["curve"][0.0]
