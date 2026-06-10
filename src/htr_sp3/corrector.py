# src/htr_sp3/corrector.py
"""The RAG corrector: repair OCR spelling errors against a vocabulary.

Pipeline per word (see the SP-3 design doc):
  1. tokenize the text into word / non-word chunks so the original spacing & punctuation can be
     rebuilt verbatim;
  2. a word already in the vocabulary is kept as-is (idempotent on correct words);
  3. an out-of-vocabulary word is vectorized and the store returns the cosine-nearest candidates;
  4. those candidates are reranked by NORMALIZED LEVENSHTEIN distance (cosine screens, edit
     distance decides — it is the better judge of "same word, mistyped");
  5. the best candidate replaces the word ONLY if its distance <= threshold; otherwise the
     original word is kept (protects proper nouns and genuine OOV);
  6. the replacement inherits the original token's capitalization.

The corrector depends only on a `VectorStore` (for retrieval) and a `vocab` set (for the
exact-match gate), so it is fully testable with InMemoryVectorStore and no database.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from . import config, vectorize
from .store import VectorStore

# Split text into alternating word / non-word chunks, keeping BOTH so we can rejoin losslessly.
# A word allows intra-word apostrophes ("don't"); everything else (spaces, punctuation, digits)
# is a non-word chunk that passes through untouched.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^A-Za-z]+")


def _normalized_levenshtein(a: str, b: str) -> float:
    """Levenshtein edit distance between *a* and *b*, divided by the longer length (0..1)."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _match_case(original: str, replacement: str) -> str:
    """Apply *original*'s capitalization pattern to the lowercase *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


class RagCorrector:
    """Correct OCR text using vocabulary retrieval + edit-distance rerank."""

    def __init__(self, store: VectorStore, vocab: Set[str], threshold: float = config.DEFAULT_THRESHOLD) -> None:
        """
        Args:
            store:     populated VectorStore (in-memory for tests, pgvector in production).
            vocab:     set of valid lowercased words for the exact-match gate.
            threshold: max normalized Levenshtein distance to accept a correction (0..1).
        """
        self._store = store
        self._vocab = vocab
        self._threshold = threshold

    def correct(self, text: str) -> Tuple[str, List[Dict[str, object]]]:
        """Return (corrected_text, log) where log lists {from, to, distance} per replacement."""
        out: List[str] = []
        log: List[Dict[str, object]] = []

        for chunk in _TOKEN_RE.findall(text):
            # Non-word chunk (spaces/punctuation/digits) -> pass through unchanged.
            if not chunk[:1].isalpha():
                out.append(chunk)
                continue

            lower = chunk.lower()
            # Already a valid word -> keep it (no correction).
            if lower in self._vocab:
                out.append(chunk)
                continue

            best = self._best_candidate(lower)
            if best is not None and best[1] <= self._threshold:
                word, distance = best
                out.append(_match_case(chunk, word))
                log.append({"from": lower, "to": word, "distance": round(distance, 4)})
            else:
                out.append(chunk)  # too far / no candidate -> leave original

        return "".join(out), log

    def _best_candidate(self, word: str) -> Tuple[str, float] | None:
        """Return the (candidate, normalized_levenshtein) with the smallest edit distance among
        the cosine-nearest neighbours, or None if the store is empty.
        """
        hits = self._store.nearest(vectorize.word_to_vector(word), k=config.K_NEIGHBORS)
        if not hits:
            return None
        ranked = [(cand, _normalized_levenshtein(word, cand)) for cand, _cos in hits]
        ranked.sort(key=lambda pair: pair[1])
        return ranked[0]
