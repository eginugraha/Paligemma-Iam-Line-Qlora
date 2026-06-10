# src/htr_sp3/vectorize.py
"""Turn a word into a fixed-length character-trigram vector for cosine retrieval.

Approach (classic spelling-similarity vector):
  1. lowercase + pad the word with boundary markers so prefixes/suffixes get their own n-grams,
  2. slice into character n-grams,
  3. feature-hash each n-gram into one of VECTOR_DIM buckets (hashlib -> deterministic across
     processes, unlike Python's salted hash()),
  4. L2-normalize so cosine similarity == dot product.

Why hashing: the raw trigram space is huge and sparse; hashing gives a small dense fixed-size
vector that pgvector can index, while preserving "shares many trigrams => high cosine".
"""
from __future__ import annotations

import hashlib
from typing import List

from . import config

# Boundary marker added around a word so that, e.g., the leading "me" of "medical" becomes a
# distinct trigram ("#me") from a mid-word "me". One marker per side is enough for trigrams.
_PAD = "#"


def _ngrams(word: str, n: int) -> List[str]:
    """Return the character n-grams of *word* after boundary padding.

    Args:
        word: already-lowercased word.
        n: n-gram size (config.NGRAM_N).

    Returns:
        List of n-character substrings; empty if the word is empty.
    """
    if not word:
        return []
    padded = _PAD * (n - 1) + word + _PAD * (n - 1)
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def _bucket(ngram: str) -> int:
    """Hash an n-gram to a stable bucket index in [0, VECTOR_DIM).

    Uses md5 (via hashlib) rather than the builtin hash() because hash() is randomized per
    process (PYTHONHASHSEED), which would make vectors differ between ingest and query runs.
    """
    digest = hashlib.md5(ngram.encode("utf-8")).hexdigest()
    return int(digest, 16) % config.VECTOR_DIM


def word_to_vector(word: str) -> List[float]:
    """Vectorize a single word into a fixed-length, L2-normalized list of floats.

    Args:
        word: any string; case is ignored.

    Returns:
        A list of length config.VECTOR_DIM. All zeros for an empty word.
    """
    vec = [0.0] * config.VECTOR_DIM
    for ng in _ngrams(word.lower(), config.NGRAM_N):
        vec[_bucket(ng)] += 1.0

    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec  # empty word -> zero vector (callers treat this as "no signal")
    return [x / norm for x in vec]
