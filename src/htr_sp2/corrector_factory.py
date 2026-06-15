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
        from htr_sp3.english_words import load_english_words
        from htr_sp3.store import PgVectorStore
        from htr_sp3.vocab import build_gate_vocabulary

        # Gate = IAM-train words UNION a general English wordlist (Option B). Widening only the
        # gate (not the candidate store) stops valid English words absent from IAM-train from
        # being treated as OOV and over-corrected — see docs/sp3-rag-correction-investigation-*.
        # The candidate STORE stays train-only (PgVectorStore populated by ingest_sp3.py), so
        # this introduces no leakage: a general dictionary is external knowledge, not test labels.
        vocab = build_gate_vocabulary(data.load_iam_splits()["train"], load_english_words())
        _CORRECTOR = RagCorrector(
            store=PgVectorStore(),
            vocab=vocab,
            threshold=sp3config.DEFAULT_THRESHOLD,
            possessive_aware=True,  # leave possessives/contractions of real words untouched
        )
        _BUILT = True

    return _CORRECTOR
