# src/htr_sp3/store.py
"""Vector storage for the vocabulary, behind a swappable interface.

`VectorStore` is a Protocol (structural type) so the corrector depends on the behaviour, not a
concrete class — exactly the InferenceEngine/FakeEngine pattern from SP-2. Two implementations:

  - InMemoryVectorStore : pure-numpy cosine; used by the whole test suite and small local runs,
                          so NO PostgreSQL is needed to develop or test SP-3.
  - PgVectorStore       : PostgreSQL + pgvector for production (added in a later task).
"""
from __future__ import annotations

from typing import List, Protocol, Tuple

# A retrieval hit: (word, cosine_distance) where distance = 1 - cosine_similarity (0 == identical).
Hit = Tuple[str, float]
# An ingest row: (word, vector).
Row = Tuple[str, List[float]]


class VectorStore(Protocol):
    """Behaviour every store must provide."""

    def add_many(self, rows: List[Row]) -> None:
        """Insert (word, vector) rows."""
        ...

    def nearest(self, vector: List[float], k: int) -> List[Hit]:
        """Return the k nearest words to *vector* as (word, distance), ascending by distance."""
        ...


class InMemoryVectorStore:
    """Reference VectorStore using numpy. Vectors are assumed L2-normalized (vectorize does this),
    so cosine similarity is a plain dot product and distance is 1 - dot.
    """

    def __init__(self) -> None:
        self._words: List[str] = []
        self._matrix = None  # lazily built numpy array of shape (N, dim)

    def add_many(self, rows: List[Row]) -> None:
        import numpy as np

        for word, vec in rows:
            self._words.append(word)
        new = np.array([vec for _, vec in rows], dtype="float32")
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    def nearest(self, vector: List[float], k: int) -> List[Hit]:
        import numpy as np

        if self._matrix is None or len(self._words) == 0:
            return []
        q = np.array(vector, dtype="float32")
        sims = self._matrix @ q            # dot product == cosine (vectors are normalized)
        distances = 1.0 - sims
        order = np.argsort(distances)[:k]  # ascending distance
        return [(self._words[i], float(distances[i])) for i in order]
