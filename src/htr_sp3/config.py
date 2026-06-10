"""Central configuration for SP-3 (RAG correction).

Every tunable number lives here so the rest of the code never hard-codes values. Matches the
htr_sp1.config philosophy: change a hyperparameter in ONE place and the whole pipeline follows.
"""
from __future__ import annotations

import os

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
# Levenshtein distance is <= this. 0.34 ~= "at most ~1 edit per 3 characters". `tune.py`
# overrides this with the value that minimizes validation CER.
DEFAULT_THRESHOLD = 0.34

# --- Storage ----------------------------------------------------------------------------

# PostgreSQL connection string for the production store. Kept in the environment so no
# credentials live in the repo (same pattern as SP-1/SP-2 env config).
PG_DSN = os.environ.get("HTR_PG_DSN", "postgresql://localhost:5432/htr")

# Table name for the vocabulary vectors.
VOCAB_TABLE = "iam_vocab"
