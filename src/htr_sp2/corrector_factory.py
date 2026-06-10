"""Build the optional M3/M4 RAG corrector for detect_stream — or None.

This is the corrector analogue of htr_sp2.engine.get_engine(): api.py asks for a corrector and
gets either None (RAG off -> M1/M2 only, backward compatible) or a ready RagCorrector. The
corrector is built ONCE and cached for the process lifetime — the IAM-train vocabulary load is
the only heavy step; the PgVectorStore connects lazily per query, so a dead/empty DB surfaces as
isolated m3/m4 error events inside detect_stream rather than a startup crash.

All heavy imports (htr_sp1.data, htr_sp3.*) are LOCAL to the function so importing this module
stays cheap and unit tests never pull datasets, torch, or a DB driver.
"""
from __future__ import annotations

from htr_sp2 import config

# Process-lifetime cache. Built on first use when RAG is enabled.
_CORRECTOR = None
_BUILT = False


def get_corrector():
    """Return a cached RagCorrector when RAG is enabled, else None.

    Returns None whenever config.ENABLE_RAG is false (checked every call, so the flag can be
    toggled in tests). When enabled, builds a RagCorrector backed by PgVectorStore (reads
    HTR_PG_DSN) and the IAM-train vocabulary, and caches it.
    """
    global _CORRECTOR, _BUILT

    if not config.ENABLE_RAG:
        return None

    if not _BUILT:
        # Local (lazy) imports: keep module import cheap; tests mock these.
        from htr_sp1 import data
        from htr_sp3 import config as sp3config
        from htr_sp3.corrector import RagCorrector
        from htr_sp3.store import PgVectorStore
        from htr_sp3.vocab import build_vocabulary

        # Vocabulary = the exact-match gate (valid words are left untouched). Train split only,
        # matching scripts/ingest_sp3.py (anti-leakage). This is the only heavy step.
        vocab = build_vocabulary(data.load_iam_splits()["train"])
        _CORRECTOR = RagCorrector(
            store=PgVectorStore(),
            vocab=vocab,
            threshold=sp3config.DEFAULT_THRESHOLD,
        )
        _BUILT = True

    return _CORRECTOR
