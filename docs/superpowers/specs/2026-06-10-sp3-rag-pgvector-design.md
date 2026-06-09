# SP-3 — RAG / pgvector Correction (M3 & M4) — Design

**Date:** 2026-06-10
**Status:** Approved (design)
**Depends on:** SP-1 (model + `htr_sp1.metrics`, `htr_sp1.data`), SP-2 (orchestrator `detect_stream`, `schemas`, `config`).
**Blocks:** SP-4 (Svelte frontend), SP-5 (batch eval + dashboard).

---

## 1. Purpose

SP-3 adds a **lexical correction layer** over the OCR output, implementing two of the
thesis's four scenarios:

- **M3 (QLoRA + RAG):** correct the raw transcription from **M1**.
- **M4 (Hybrid CoT + RAG):** correct the parsed transcription from **M2** (CoT).

RAG here operates on **text, not images** (per PRD M3). The handwriting image is already
turned into text by PaliGemma (M1/M2); SP-3 only repairs OCR spelling errors against a
vocabulary of valid English words (e.g. `medisal → medical`). No images enter this stage.

## 2. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Scope | M3 **and** M4 |
| Retrieval/correction method | char n-gram (n=3) feature-hashed to a fixed **D=512** vector + pgvector cosine to retrieve candidates, then **Levenshtein** rerank to decide |
| Vocabulary source | unique words from IAM **train** split only (anti–test-leakage) |
| Correction policy | conservative + gated; threshold **tuned on the validation split** (minimize CER) |
| Architecture | corrector as a post-processor in the orchestrator; swappable `VectorStore` (`PgVectorStore` + `InMemoryVectorStore`) mirroring SP-2's `InferenceEngine`/`FakeEngine` |

## 3. Architecture & components

New package `htr_sp3` (parallel to `htr_sp1`, `htr_sp2`), small single-purpose units:

```
src/htr_sp3/
  config.py        # n-gram size (n=3), vector dim D=512, k-neighbors, default threshold, env HTR_PG_DSN
  vocab.py         # build_vocabulary(train_records) -> set[str]  (lowercased, dedup, filtered)
  vectorize.py     # word_to_vector(word) -> list[float]  (char n-gram -> feature-hashed to D=512, L2-normalized)
  store.py         # VectorStore Protocol: add_many(items), nearest(vector, k) -> [(word, distance)]
                   #   PgVectorStore        : Postgres + pgvector, cosine (<=>)
                   #   InMemoryVectorStore  : numpy cosine; tests + small local runs
  corrector.py     # RagCorrector(store, threshold): correct(text) -> (corrected_text, log)
  ingest.py        # build vocab -> vectorize -> store.add_many -> build index
  tune.py          # find threshold minimizing CER on validation
scripts/
  ingest_sp3.py    # thin CLI: populate the DB once
  tune_sp3.py      # thin CLI: scan thresholds, write tune_sp3.json
```

Responsibility boundaries: `vectorize` knows nothing about the store; `store` knows nothing
about Levenshtein; `corrector` combines them + applies the threshold gate. Each unit is
understandable and testable in isolation.

## 4. Correction data flow (`RagCorrector.correct`)

Given one transcription string (M1 or M2 output), return `(corrected_text, log)`:

1. **Tokenize** into word tokens + separators (regex) so the original string can be rebuilt
   verbatim (whitespace/punctuation preserved).
2. For each **word** token:
   a. normalize (lowercase, strip surrounding punctuation) for lookup;
   b. if already in vocab → keep the original token unchanged (idempotent on valid words);
   c. if OOV → `vectorize(word)` → `store.nearest(vec, k=5)`;
   d. rerank the k candidates by **normalized Levenshtein** distance vs the word;
   e. take the best candidate; replace **only if** its distance ≤ threshold `T`; else keep
      the original token (protects proper nouns / true OOV);
   f. restore the original token's capitalization & punctuation onto the replacement.
3. **Reconstruct** the string from tokens + original separators.
4. Build a structured **log**: list of `{from, to, distance}` corrections.

Notes:
- Cosine (pgvector) **screens** candidates; Levenshtein **decides** — both surfaced in the log
  for transparency.
- Reported `distance` = normalized Levenshtein (the value the gate uses).

## 5. Orchestrator integration (M3/M4)

`detect_stream` currently runs `_SPECS = [m1, m2]` via `engine.run`. M3/M4 are **not** engine
calls — they correct text already produced. Plan:

- Add an optional `corrector` parameter to `detect_stream`.
  - `corrector is None` (e.g. DB unavailable, legacy tests) → M3/M4 are skipped cleanly →
    backward-compatible with the 41 existing SP-2 tests.
  - `corrector` present → after M1/M2, emit M3 (correct M1 text) and M4 (correct M2 text).
- Keep the M1/M2 result text in local variables; pass them to `corrector.correct(...)`.
- **Failure isolation** preserved: wrap each correction in try/except → a failing M3 emits
  `error_event("m3", ...)` and the stream continues (same pattern as M1/M2).
- **M4 dependency:** if M2 failed (no text), skip M4 with
  `error_event("m4", "depends on m2 which failed")`.
- **Reuse `schemas.result_event`** (already has model/text/cer/wer/latency/log/status_tag):
  - `model`: `"m3"` / `"m4"`
  - `text`: corrected text
  - `cer`/`wer`: recomputed vs ground_truth via `htr_sp1.metrics` (the key thesis numbers)
  - `log`: e.g. `"RAG: medisal→medical (0.13), recyrd→record (0.17)"`
  - `status_tag`: `"Corrected"` (M3) / `"Optimal"` (M4) per PRD
- Add `M3_STATUS_TAG="Corrected"`, `M4_STATUS_TAG="Optimal"` to `htr_sp2.config`.

## 6. pgvector storage & ingestion

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE iam_vocab (
    word TEXT PRIMARY KEY,   -- valid word (lowercased), unique
    vec  vector(512)         -- char n-gram feature-hashed to D=512 (config), L2-normalized
);
CREATE INDEX ON iam_vocab USING hnsw (vec vector_cosine_ops);  -- after bulk insert
```

- Nearest query: `SELECT word, vec <=> %s AS distance FROM iam_vocab ORDER BY vec <=> %s LIMIT k;`
- Driver `psycopg` (v3); DSN via env `HTR_PG_DSN` (no credentials in code).
- Vectors L2-normalized at ingest for consistent cosine; HNSW index (vocab ~tens of thousands
  of words — light for an 8GB Mac).
- Ingestion (`ingest.py`): load IAM train → `build_vocabulary` → vectorize → `add_many` →
  build index. Idempotent (truncate/upsert) so it can be rebuilt without duplicates.

## 7. Testing strategy (no Postgres in CI)

- `InMemoryVectorStore` satisfies the same `VectorStore` Protocol → the corrector is fully
  tested without a database.
- Unit tests (SP-1/SP-2 style):
  - `test_sp3_vectorize` — n-grams correct, deterministic, L2-normalized.
  - `test_sp3_store` — InMemory `nearest` returns correct distance ordering.
  - `test_sp3_vocab` — dedup/normalize; built from **train only** (leakage guard).
  - `test_sp3_corrector` — valid word untouched; near-OOV corrected; far-OOV (> T) left;
    case/punctuation restored; log correct.
  - `test_sp3_orchestrator` — `detect_stream` with a fake corrector emits m3/m4; without a
    corrector they are skipped; a correction error yields `error_event` and the stream
    continues.
- `PgVectorStore` is **not** exercised in CI (needs a DB); verified manually via the ingest
  script + a smoke query. One optional test that is skipped when `HTR_PG_DSN` is unset.

## 8. Threshold tuning & evaluation

`tune.py` / `scripts/tune_sp3.py`: find `T` minimizing **CER on validation** (not test).

1. Vocab store populated from IAM train (ingest done).
2. Correction input = M1 model predictions on validation (from a saved predictions file, or
   produced by running M1). The tuning code + tests can be written now with a fake
   corrector/store; the real scan runs after the model exists (PaliGemma re-train is the day
   after this design, 2026-06-11) — **SP-3 is not blocked by training.**
3. For each candidate `T` in a linear grid (e.g. 0.10…0.50): correct all samples, compute
   mean CER via `htr_sp1.metrics`.
4. Pick the lowest-CER `T`; write `tune_sp3.json` + print the CER-vs-T curve.
5. Also report the **no-correction baseline** (corrector off) so the RAG gain is visible —
   direct material for Chapter 4.

Official M3/M4 numbers are computed on **test** with the tuned `T`; the full batch run +
dashboard belong to SP-5, which reuses this corrector and metrics.

## 9. Out of scope (deferred)

- General English dictionary / IAM+dictionary union vocab (future lever).
- Local GGUF inference engine (separate sub-project).
- Svelte frontend (SP-4); batch evaluation + dashboard (SP-5).
- Semantic/neural embeddings (wrong tool for spelling correction; rejected).
