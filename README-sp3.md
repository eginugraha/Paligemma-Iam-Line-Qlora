# SP-3 — RAG / pgvector Correction (M3 & M4)

A **lexical corrector** over the OCR text produced by SP-1/SP-2. It repairs spelling
errors word-by-word against a vocabulary of valid words (e.g. `medisal → medical`). This
is text-only RAG — no images enter this stage. It implements two thesis scenarios:

- **M3 (QLoRA + RAG):** correct the raw transcription from **M1**.
- **M4 (Hybrid CoT + RAG):** correct the parsed transcription from **M2** (CoT).

## How a word is corrected

1. **Vocabulary gate** — a word already in the IAM-**train** vocabulary is left untouched.
2. **Vectorize** the out-of-vocabulary word: character trigrams → feature-hashed to a fixed
   `D=512` vector → L2-normalized (`htr_sp3.vectorize`).
3. **Screen** with cosine: the `VectorStore` returns the `k` nearest vocabulary words.
4. **Decide** with a normalized **Levenshtein** rerank — the closest edit-distance candidate
   wins, but only if its distance ≤ a tuned **threshold**; otherwise the original word is kept
   (protects proper nouns / true OOV). Capitalization and punctuation are preserved.

Cosine *screens*, edit distance *decides*; both are surfaced in the correction log.

## Storage backends (swappable `VectorStore`)

- **`InMemoryVectorStore`** — numpy cosine. Used by the whole test suite and small local runs,
  so **no PostgreSQL is needed to develop or test SP-3**.
- **`PgVectorStore`** — PostgreSQL + the `pgvector` extension (cosine `<=>`, HNSW index) for
  production. Both stores tie-break candidates by word, so the in-memory store (used for
  threshold tuning) and pgvector (production) return identical candidate sets.

## Orchestrator wiring

`htr_sp2.detect_stream` gains an optional `corrector` parameter. When supplied, it emits
`m3` (corrected M1) and `m4` (corrected M2) after M1/M2, with CER/WER recomputed via
`htr_sp1.metrics`. When `None`, behaviour is unchanged (backward-compatible with the SP-2
stream). Each correction is failure-isolated; M4 is skipped if M2 produced no text.

## Tests

    pytest -q

All SP-3 tests run on CPU with no database (`InMemoryVectorStore`). The pgvector path has one
opt-in test that is **skipped** unless `HTR_PG_DSN` is set.

## Production setup (needs a DB + the re-trained model)

These run outside the test loop:

    # 1. Start PostgreSQL with pgvector, then point SP-3 at it
    export HTR_PG_DSN="postgresql://user:pass@localhost:5432/htr"

    # 2. Build the IAM-train vocabulary and load it into pgvector (idempotent)
    python scripts/ingest_sp3.py

    # 3. Tune the correction threshold on VALIDATION M1 predictions (min CER)
    #    --pairs is a JSON list of {"prediction": ..., "ground_truth": ...}
    python scripts/tune_sp3.py --pairs val_m1_predictions.json --out tune_sp3.json

    # 4. Verify the live DB path once
    HTR_PG_DSN=... pytest tests/test_sp3_store.py -k pgvector

Then set the tuned value as `htr_sp3.config.DEFAULT_THRESHOLD` and pass a real `RagCorrector`
into the SP-2 API call site to activate M3/M4 in production.

## Scope

M3 + M4 lexical correction only. A general-English dictionary vocabulary, the Svelte frontend
(SP-4), and the batch evaluation + dashboard (SP-5, which reuses this corrector) are separate
sub-projects. Design: `docs/superpowers/specs/2026-06-10-sp3-rag-pgvector-design.md`.
