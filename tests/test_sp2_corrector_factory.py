"""get_corrector() is the optional-RAG analogue of get_engine(): None when RAG is off
(default), else a cached RagCorrector backed by PgVectorStore + IAM-train vocab. We mock the
dataset load and the DB store so the test needs neither downloads nor Postgres.
"""
from htr_sp2 import config, corrector_factory


def _reset_cache():
    # The factory caches the built corrector at module level; reset between cases.
    corrector_factory._CORRECTOR = None
    corrector_factory._BUILT = False


def test_returns_none_when_rag_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RAG", False)
    _reset_cache()
    assert corrector_factory.get_corrector() is None


def test_builds_pgvector_rag_when_enabled(monkeypatch):
    from htr_sp3.corrector import RagCorrector

    monkeypatch.setattr(config, "ENABLE_RAG", True)
    _reset_cache()

    # Fake IAM train -> no dataset download; the vocab is built from this text.
    import htr_sp1.data as sp1data
    monkeypatch.setattr(sp1data, "load_iam_splits",
                        lambda: {"train": [{"text": "the medical record"}]})

    # Fake PgVectorStore -> no real DB connection (it is never queried in this test).
    import htr_sp3.store as sp3store
    monkeypatch.setattr(sp3store, "PgVectorStore", lambda *a, **k: object())

    corrector = corrector_factory.get_corrector()
    assert isinstance(corrector, RagCorrector)
    # The vocab gate was built from the fake train split: a valid word is left untouched
    # WITHOUT touching the store (so the fake store is never queried).
    text, log = corrector.correct("the")
    assert text == "the" and log == []


def test_corrector_is_cached(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_RAG", True)
    _reset_cache()

    import htr_sp1.data as sp1data
    calls = {"n": 0}

    def _counting_load():
        calls["n"] += 1
        return {"train": [{"text": "the medical record"}]}

    monkeypatch.setattr(sp1data, "load_iam_splits", _counting_load)
    import htr_sp3.store as sp3store
    monkeypatch.setattr(sp3store, "PgVectorStore", lambda *a, **k: object())

    first = corrector_factory.get_corrector()
    second = corrector_factory.get_corrector()
    assert first is second              # same cached instance
    assert calls["n"] == 1              # vocab/dataset built only once
