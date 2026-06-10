# SP-2 — Wire RAG corrector (M3/M4) + CORS into the API — Design

**Date:** 2026-06-11
**Status:** Approved (design)
**Depends on:** SP-2 (`api.py`, `orchestrator.detect_stream`, `engine.get_engine`, `config`), SP-3 (`RagCorrector`, `PgVectorStore`, `build_vocabulary`), SP-1 (`data.load_iam_splits`).
**Enables:** SP-4 (Svelte frontend) — lets the live API stream all four scenarios (M1–M4) and be called from a browser on another origin.

---

## 1. Purpose

Two small gaps block a complete SP-4 frontend:

1. `api.py` calls `detect_stream(...)` **without** a corrector, so the live API never emits
   M3/M4 — only M1/M2.
2. FastAPI has **no CORS** configured, so a browser frontend on a different origin (e.g. the
   Vite/SvelteKit dev server) cannot call `/v1/detect`.

This work wires an **optional** RAG corrector into the API and adds CORS — pure plumbing. It
does **not** populate pgvector or tune the threshold (separate manual ops, see §7).

## 2. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| RAG activation | **Off by default**; opt-in via env `HTR_ENABLE_RAG` (M1/M2 behaviour unchanged when off) |
| Corrector store | **PgVectorStore only** (prod-faithful; requires a live, ingested Postgres+pgvector) |
| Factory shape | `get_corrector()` in a new `htr_sp2/corrector_factory.py`, mirroring `engine.get_engine()`; returns `None` or a cached `RagCorrector` |
| Vocabulary | IAM **train** split via `htr_sp1.data.load_iam_splits()["train"]` → `build_vocabulary` (built once, cached) |
| Threshold | `htr_sp3.config.DEFAULT_THRESHOLD` (until tuning lands) |
| DB-down behaviour | `PgVectorStore` connects lazily per query, so a dead DB surfaces as isolated m3/m4 **error events** (m1/m2 still succeed), not a startup crash |
| CORS | `CORSMiddleware`, `allow_origins = config.CORS_ORIGINS` (env `HTR_CORS_ORIGINS`, default `["*"]`, `allow_credentials=False`) |

## 3. Components

### `src/htr_sp2/config.py` (MODIFY — add two settings)

```python
# RAG (M3/M4) is OFF unless explicitly enabled. When on, the API builds a PgVectorStore-backed
# corrector (needs a live, ingested Postgres+pgvector — see scripts/ingest_sp3.py).
ENABLE_RAG = os.environ.get("HTR_ENABLE_RAG", "0").lower() in ("1", "true", "yes")

# Browser origins allowed to call the API (CORS). Comma-separated; default "*" for the
# prototype frontend. Credentials are not used, so "*" is safe.
CORS_ORIGINS = os.environ.get("HTR_CORS_ORIGINS", "*").split(",")
```

### `src/htr_sp2/corrector_factory.py` (NEW)

`get_corrector() -> RagCorrector | None`. Single responsibility: build the optional M3/M4
corrector for `detect_stream`, or return `None`.

- Returns `None` when `config.ENABLE_RAG` is false (checked every call so toggling works).
- When enabled, builds **once** and caches at module level: `RagCorrector(store=PgVectorStore(),
  vocab=build_vocabulary(load_iam_splits()["train"]), threshold=sp3config.DEFAULT_THRESHOLD)`.
- Heavy imports (`htr_sp1.data`, `htr_sp3.*`) are **local** to the function so importing the
  module stays cheap and unit tests don't pull datasets/torch.
- The vocab load is the only heavy step; `PgVectorStore` does not connect until `nearest()`.

### `src/htr_sp2/api.py` (MODIFY)

- Add `from htr_sp2 import config`, `from htr_sp2.corrector_factory import get_corrector`,
  `from fastapi.middleware.cors import CORSMiddleware`.
- After `app = FastAPI(...)`: `app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
  allow_methods=["*"], allow_headers=["*"])`.
- In `detect()`: `corrector = get_corrector()` then
  `detect_stream(engine, image, file.filename or "upload", ground_truth, corrector=corrector)`.

## 4. Data flow

`POST /v1/detect` → validate image → `engine = get_engine()`, `corrector = get_corrector()` →
`detect_stream(...)` emits `meta`, `result`/`error` for m1, m2, and (when corrector is not None)
m3, m4 → `done`. Identical NDJSON contract; only the number of scenarios changes.

## 5. Error handling

- **RAG off:** `get_corrector()` returns `None` → unchanged M1/M2 stream.
- **RAG on, DB reachable + ingested:** m3/m4 stream corrected text + recomputed CER/WER.
- **RAG on, DB down/empty:** the corrector's `PgVectorStore.nearest()` raises (or returns no
  rows) per request; `detect_stream` already isolates this into m3/m4 `error` events while m1/m2
  succeed and the stream still ends with `done`. No startup crash, no impact on M1/M2.

## 6. Testing strategy (CPU, no DB/network)

- `test_sp2_corrector_factory`:
  - default (`ENABLE_RAG` false) → `get_corrector()` is `None`.
  - enabled (`monkeypatch config.ENABLE_RAG=True`) → returns a `RagCorrector`, built from a
    **fake** IAM train (monkeypatch `htr_sp1.data.load_iam_splits`) and a **fake/stub**
    `PgVectorStore` (monkeypatch so no real DB); assert the vocab gate works. Reset the module
    cache between cases.
- `test_sp2_api` (extend): a request with an `Origin` header gets an `Access-Control-Allow-Origin`
  response header (CORS active). Existing `/v1/detect` tests still pass unchanged (corrector
  defaults to `None` → m1/m2 only).
- Full suite stays green (the live pgvector path remains opt-in / manual).

## 7. Out of scope (manual, after this lands)

To actually see M3/M4 corrections in the live API (DB is already running):
1. `python scripts/ingest_sp3.py` — populate pgvector with the IAM-train vocabulary (**required**;
   an empty DB means the corrector has nothing to retrieve).
2. `export HTR_ENABLE_RAG=1` and run the API.
3. (Optional) `scripts/tune_sp3.py` → set the tuned threshold.

Not included: SP-4 frontend itself; pgvector ingestion/tuning code (already exists in SP-3);
retraining.
