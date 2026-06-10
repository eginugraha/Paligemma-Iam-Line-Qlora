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
        sims = self._matrix @ q   # dot product == cosine similarity (vectors are L2-normalized)
        distances = 1.0 - sims
        # Sort by (distance ascending, word ascending) — the secondary word key breaks ties
        # deterministically.  Without it, np.argsort uses insertion order for equal distances,
        # which can differ from PgVectorStore's tie ordering and cause the candidate set returned
        # here to diverge from what production pgvector returns.  A threshold tuned against this
        # store must transfer faithfully to production, so both stores MUST agree on which k words
        # are selected when several vocab words land at the same cosine distance.
        order = sorted(range(len(self._words)), key=lambda i: (float(distances[i]), self._words[i]))[:k]
        return [(self._words[i], float(distances[i])) for i in order]


class PgVectorStore:
    """Production VectorStore backed by PostgreSQL + the pgvector extension.

    Cosine distance uses pgvector's `<=>` operator. Vectors are stored as the pgvector `vector`
    type; we pass them as the literal string "[v1,v2,...]" which the type accepts. psycopg v3 is
    imported lazily so the rest of SP-3 (and the test suite) never requires a DB driver.
    """

    def __init__(self, dsn: str | None = None) -> None:
        from . import config
        self._dsn = dsn or config.PG_DSN
        self._table = config.VOCAB_TABLE
        self._dim = config.VECTOR_DIM

    def _connect(self):
        import psycopg
        return psycopg.connect(self._dsn)

    @staticmethod
    def _to_literal(vector: List[float]) -> str:
        # pgvector accepts a text literal like "[0.1,0.2,...]".
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    def create_schema(self) -> None:
        """Create the pgvector extension + vocab table (idempotent). Truncates existing rows."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} "
                f"(word TEXT PRIMARY KEY, vec vector({self._dim}))"
            )
            cur.execute(f"TRUNCATE {self._table}")
            conn.commit()

    def add_many(self, rows: List[Row]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._table} (word, vec) VALUES (%s, %s) "
                f"ON CONFLICT (word) DO UPDATE SET vec = EXCLUDED.vec",
                [(word, self._to_literal(vec)) for word, vec in rows],
            )
            conn.commit()

    def create_index(self) -> None:
        """Build the HNSW cosine index after bulk insert (faster than inserting into an index)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_vec_hnsw "
                f"ON {self._table} USING hnsw (vec vector_cosine_ops)"
            )
            conn.commit()

    def nearest(self, vector: List[float], k: int) -> List[Hit]:
        lit = self._to_literal(vector)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                # The secondary `, word` sort key breaks ties by word lexicographically,
                # matching the tie-break order of InMemoryVectorStore.nearest().  Without it,
                # PostgreSQL may return a different arbitrary subset of equal-distance rows,
                # causing threshold calibration done against the in-memory store to produce
                # subtly different results in production pgvector — a reproducibility hazard.
                f"SELECT word, vec <=> %s AS distance FROM {self._table} "
                f"ORDER BY vec <=> %s, word LIMIT %s",
                (lit, lit, k),
            )
            return [(word, float(dist)) for word, dist in cur.fetchall()]
