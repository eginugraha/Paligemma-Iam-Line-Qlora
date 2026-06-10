"""RagCorrector repairs OCR text word-by-word: valid words are left alone; an OOV word is
replaced by its nearest vocabulary word ONLY when the normalized Levenshtein distance is within
the threshold. Case and punctuation of the original token are preserved.
"""
from htr_sp3 import vectorize
from htr_sp3.corrector import RagCorrector
from htr_sp3.store import InMemoryVectorStore

VOCAB = ["medical", "record", "the", "was", "patient"]


def _corrector(threshold=0.34):
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in VOCAB])
    return RagCorrector(store=store, vocab=set(VOCAB), threshold=threshold)


def test_valid_words_are_left_unchanged():
    text, log = _corrector().correct("the patient record")
    assert text == "the patient record"
    assert log == []


def test_near_oov_word_is_corrected():
    text, log = _corrector().correct("medisal recyrd")
    assert text == "medical record"
    assert {c["from"] for c in log} == {"medisal", "recyrd"}
    assert {c["to"] for c in log} == {"medical", "record"}


def test_far_oov_word_is_left_alone():
    # "xylophone" is nowhere near any vocab word -> distance exceeds threshold -> unchanged.
    text, log = _corrector().correct("xylophone")
    assert text == "xylophone"
    assert log == []


def test_capitalization_is_preserved_on_correction():
    text, _ = _corrector().correct("Medisal")
    assert text == "Medical"


def test_punctuation_and_spacing_are_preserved():
    text, _ = _corrector().correct("medisal, the record.")
    assert text == "medical, the record."


def test_threshold_zero_corrects_nothing():
    text, log = _corrector(threshold=0.0).correct("medisal")
    assert text == "medisal"
    assert log == []
