# SP-2 RAG Corrector Wiring + CORS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live `/v1/detect` API stream M3/M4 (when RAG is enabled) and be callable from a browser on another origin (CORS).

**Architecture:** Add an env-gated `get_corrector()` factory (mirroring `get_engine()`) that returns `None` by default or a cached `RagCorrector` backed by `PgVectorStore` + IAM-train vocab when `HTR_ENABLE_RAG` is set. `api.py` passes it to `detect_stream` and mounts `CORSMiddleware`. RAG off by default keeps M1/M2 behaviour (and all existing tests) unchanged.

**Tech Stack:** Python, FastAPI (`CORSMiddleware`, `TestClient`), pytest, `unittest`-style monkeypatch. SP-3 `RagCorrector`/`PgVectorStore`/`build_vocabulary`, SP-1 `data.load_iam_splits`.

**Reference spec:** `docs/superpowers/specs/2026-06-11-sp2-rag-corrector-wiring-cors-design.md`

**Conventions (match existing SP-2):** heavy module/function docstrings + inline comments (thesis must be explainable); heavy imports (`htr_sp1.data`, `htr_sp3.*`, datasets) are done lazily *inside* functions so the module imports instantly and unit tests stay on CPU with no DB/network; tests live in `tests/` and run with plain `pytest`.

---

## File Structure

```
src/htr_sp2/
  config.py             # MODIFY: add ENABLE_RAG + CORS_ORIGINS
  corrector_factory.py  # NEW: get_corrector() -> RagCorrector | None (cached, env-gated)
  api.py                # MODIFY: mount CORSMiddleware; pass get_corrector() to detect_stream
tests/
  test_sp2_config.py            # MODIFY: assert the two new settings' defaults
  test_sp2_corrector_factory.py # NEW: None when off; builds PgVector RAG when on (mocked)
  test_sp2_api.py               # MODIFY: assert CORS header (existing detect tests unchanged)
```

---

## Task 1: Config — `ENABLE_RAG` + `CORS_ORIGINS`

**Files:**
- Modify: `src/htr_sp2/config.py`
- Test: `tests/test_sp2_config.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_sp2_config.py`)

```python
def test_rag_is_disabled_by_default():
    # M3/M4 must be OFF unless explicitly enabled, so M1/M2 behaviour is unchanged.
    from htr_sp2 import config
    assert config.ENABLE_RAG is False


def test_cors_origins_default_is_wildcard():
    from htr_sp2 import config
    assert config.CORS_ORIGINS == ["*"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp2_config.py -k "rag_is_disabled or cors_origins" -v`
Expected: FAIL with `AttributeError: module 'htr_sp2.config' has no attribute 'ENABLE_RAG'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/htr_sp2/config.py`:

```python
# ---------------------------------------------------------------------------
# RAG correction (M3/M4) and CORS — added for the SP-4 frontend
# ---------------------------------------------------------------------------

# M3/M4 RAG correction is OFF unless explicitly enabled. When on, the API builds a
# PgVectorStore-backed corrector, which needs a live, ingested Postgres+pgvector
# (run scripts/ingest_sp3.py first). Off by default so M1/M2 behaviour is unchanged.
ENABLE_RAG = os.environ.get("HTR_ENABLE_RAG", "0").lower() in ("1", "true", "yes")

# Browser origins allowed to call the API (CORS), comma-separated. Defaults to "*" for the
# prototype frontend; credentials are not used so the wildcard is safe.
CORS_ORIGINS = os.environ.get("HTR_CORS_ORIGINS", "*").split(",")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp2_config.py -v`
Expected: PASS (the two new tests plus the existing config tests)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/config.py tests/test_sp2_config.py
git commit -m "feat(sp2): config flags ENABLE_RAG + CORS_ORIGINS"
```

---

## Task 2: `get_corrector()` factory

**Files:**
- Create: `src/htr_sp2/corrector_factory.py`
- Test: `tests/test_sp2_corrector_factory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp2_corrector_factory.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp2_corrector_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp2.corrector_factory'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp2/corrector_factory.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp2_corrector_factory.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/corrector_factory.py tests/test_sp2_corrector_factory.py
git commit -m "feat(sp2): get_corrector factory — cached PgVector RAG corrector, env-gated"
```

---

## Task 3: Wire corrector + CORS into `api.py`

**Files:**
- Modify: `src/htr_sp2/api.py`
- Test: `tests/test_sp2_api.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_sp2_api.py`)

```python
def test_cors_header_present_for_browser_origin():
    # A browser request carries an Origin header; CORSMiddleware must echo an
    # Access-Control-Allow-Origin so a frontend on another origin can read the response.
    resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.headers["access-control-allow-origin"] == "*"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp2_api.py::test_cors_header_present_for_browser_origin -v`
Expected: FAIL with `KeyError: 'access-control-allow-origin'`

- [ ] **Step 3a: Mount CORS on the app**

In `src/htr_sp2/api.py`, add these imports near the existing FastAPI imports (top of the import
block, with the other `from htr_sp2 ...` and fastapi imports):

```python
from fastapi.middleware.cors import CORSMiddleware

from htr_sp2 import config
from htr_sp2.corrector_factory import get_corrector
```

Immediately AFTER the `app = FastAPI(...)` block (i.e. after its closing `)`), add:

```python
# CORS so the SP-4 browser frontend (a different origin, e.g. the Vite dev server) can call
# the API. Credentials are not used, so a wildcard origin is safe. Origins come from config
# (env HTR_CORS_ORIGINS, default "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 3b: Pass the corrector to detect_stream**

In `src/htr_sp2/api.py`, inside the `detect(...)` handler, replace the stream-construction line:

```python
    stream = detect_stream(engine, image, file.filename or "upload", ground_truth)
```

with:

```python
    # get_corrector() returns None unless HTR_ENABLE_RAG is set (then a cached PgVector-backed
    # RagCorrector). When present, detect_stream additionally emits m3 (corrected m1) and m4
    # (corrected m2); when None, the stream is M1/M2 only — unchanged behaviour.
    corrector = get_corrector()
    stream = detect_stream(
        engine, image, file.filename or "upload", ground_truth, corrector=corrector
    )
```

- [ ] **Step 4: Run the targeted test + existing API tests (no regressions)**

Run: `python -m pytest tests/test_sp2_api.py -v`
Expected: PASS — the new CORS test passes; the existing detect tests are unchanged (RAG is off
by default, so `get_corrector()` returns None and the stream is still `meta, result, result, done`).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (all green; the opt-in pgvector DB test stays skipped).

- [ ] **Step 6: Commit**

```bash
git add src/htr_sp2/api.py tests/test_sp2_api.py
git commit -m "feat(sp2): emit M3/M4 via get_corrector + enable CORS on /v1/detect"
```

---

## Post-implementation (manual — DB is already running)

To see M3/M4 corrections from the live API:

1. `python scripts/ingest_sp3.py` — populate pgvector with the IAM-train vocabulary (**required**;
   an empty DB means the corrector retrieves nothing and corrects nothing).
2. `export HTR_ENABLE_RAG=1` then start the API (`uvicorn htr_sp2.api:app --app-dir src`).
3. A `/v1/detect` request now streams `meta, result(m1), result(m2), result(m3), result(m4), done`.
4. (Optional) `scripts/tune_sp3.py` → set the tuned threshold.

---

## Self-Review

- **Spec coverage:** §2 activation gate `HTR_ENABLE_RAG` (Task 1 config + Task 2 factory); §2/§3
  PgVectorStore-only corrector + IAM-train vocab + cache (Task 2); §3 api.py corrector pass-through
  + CORSMiddleware from `config.CORS_ORIGINS` (Task 3); §5 DB-down isolation is inherited from
  `detect_stream`'s existing per-scenario try/except (no new code, exercised manually); §6 testing
  — None-when-off, builds-when-on (mocked), cached, CORS header, existing tests unchanged (Tasks
  1–3). All spec sections map to a task.
- **Placeholders:** none — every step ships runnable code/tests and exact commands.
- **Type consistency:** `config.ENABLE_RAG: bool`, `config.CORS_ORIGINS: list[str]`,
  `get_corrector() -> RagCorrector | None` with module cache `_CORRECTOR`/`_BUILT`, and
  `detect_stream(engine, image, filename, ground_truth, corrector=...)` (the existing optional
  param) are used consistently across config.py, corrector_factory.py, and api.py.
```
