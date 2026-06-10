# src/htr_sp3/ingest.py
"""Populate a VectorStore with the IAM vocabulary.

`ingest_vocabulary` is store-agnostic (takes any VectorStore) so it is unit-tested with the
in-memory store and reused by the CLI with PgVectorStore. The CLI wiring (load IAM, build the
pg store, create schema/index) lives in scripts/ingest_sp3.py.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

from . import vectorize, vocab
from .store import VectorStore


def ingest_vocabulary(records: Iterable[Dict[str, Any]], store: VectorStore) -> int:
    """Build the vocabulary from *records* and load (word, vector) rows into *store*.

    Args:
        records: IAM TRAIN split records ({"text": ...}). Train only — see vocab.build_vocabulary.
        store:   a VectorStore to populate.

    Returns:
        Number of unique words ingested.
    """
    words = sorted(vocab.build_vocabulary(records))  # sorted -> deterministic ingest order
    rows = [(w, vectorize.word_to_vector(w)) for w in words]
    store.add_many(rows)
    return len(rows)
