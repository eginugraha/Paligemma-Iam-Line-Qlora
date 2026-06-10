# tests/test_sp3_vectorize.py
"""word_to_vector turns a word into a fixed-length, deterministic, L2-normalized vector built
from its character trigrams. Similar spellings must produce similar (high-cosine) vectors.
"""
import math

from htr_sp3 import config, vectorize


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized, so dot == cosine


def test_vector_has_fixed_dimension():
    assert len(vectorize.word_to_vector("medical")) == config.VECTOR_DIM


def test_vector_is_deterministic_across_calls():
    # Must not depend on Python's per-process hash randomization (we use hashlib, not hash()).
    assert vectorize.word_to_vector("record") == vectorize.word_to_vector("record")


def test_vector_is_l2_normalized():
    v = vectorize.word_to_vector("handwriting")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_similar_spellings_are_closer_than_dissimilar():
    medical = vectorize.word_to_vector("medical")
    medisal = vectorize.word_to_vector("medisal")   # one-letter OCR error
    zebra = vectorize.word_to_vector("zebra")       # unrelated
    assert _cosine(medical, medisal) > _cosine(medical, zebra)


def test_empty_word_returns_zero_vector():
    v = vectorize.word_to_vector("")
    assert len(v) == config.VECTOR_DIM
    assert all(x == 0.0 for x in v)
