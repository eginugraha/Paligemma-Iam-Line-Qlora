"""Central configuration for SP-3 (RAG correction).

Every tunable number lives here so the rest of the code never hard-codes values. Matches the
htr_sp1.config philosophy: change a hyperparameter in ONE place and the whole pipeline follows.
"""
from __future__ import annotations

import os

# Load a local .env (optional) before reading os.environ, so the DB credential HTR_PG_DSN can
# live in a gitignored .env at the repo root instead of being exported in every shell. We load
# it here, at the top of config, so the os.environ.get below already sees it. load_dotenv does
# not override real shell exports, and python-dotenv is optional (falls back to the shell env).
try:
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # parents[2] == repo root
except ImportError:
    pass

# --- Character-vector shape -------------------------------------------------------------

# Character n-gram size. Trigrams (n=3) balance specificity vs. robustness for spelling
# correction: long enough to be discriminative, short enough to survive a single typo.
NGRAM_N = 3

# Fixed vector dimension stored in pgvector. The raw trigram space (~26^3) is far larger than
# pgvector's practical index limit, so we feature-hash n-grams into this many buckets.
VECTOR_DIM = 512

# --- Retrieval / correction -------------------------------------------------------------

# Candidates cosine retrieves from the store before the Levenshtein rerank picks the winner.
K_NEIGHBORS = 5

# Default correction gate: an OOV word is replaced only if its best candidate's normalized
# Levenshtein distance is <= this. Tuned on 100 IAM-validation M1 predictions with the Option B
# (train ∪ English) gate: CER is ≈flat for T<=0.15 (best is technically T=0.0). We use 0.15 — it
# stays at baseline accuracy while still letting a few genuine corrections fire, so M3/M4 are not
# byte-identical to M1/M2. See docs/sp3-rag-correction-fix-2026-06-15.md and reports/tune_sp3_english.json.
DEFAULT_THRESHOLD = 0.15

# --- Storage ----------------------------------------------------------------------------

# PostgreSQL connection string for the production store. Kept in the environment so no
# credentials live in the repo (same pattern as SP-1/SP-2 env config).
PG_DSN = os.environ.get("HTR_PG_DSN", "postgresql://localhost:5432/htr")

# Table name for the vocabulary vectors.
VOCAB_TABLE = "iam_vocab"
