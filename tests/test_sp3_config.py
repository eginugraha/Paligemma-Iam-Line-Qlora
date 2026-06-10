"""Config holds the fixed RAG hyperparameters in one place (DRY), mirroring htr_sp1.config."""
from htr_sp3 import config


def test_config_has_sane_rag_constants():
    # n-gram size for character vectors; 3 (trigrams) is the spelling-correction sweet spot.
    assert config.NGRAM_N == 3
    # Fixed vector dimension for pgvector (raw trigram space is too large -> feature hashing).
    assert config.VECTOR_DIM == 512
    # How many candidate words cosine retrieves before the Levenshtein rerank.
    assert config.K_NEIGHBORS >= 1
    # Default correction gate (normalized Levenshtein, 0..1) before tuning overrides it.
    assert 0.0 < config.DEFAULT_THRESHOLD < 1.0
