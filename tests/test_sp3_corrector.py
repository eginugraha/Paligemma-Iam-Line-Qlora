"""RagCorrector repairs OCR text word-by-word: valid words are left alone; an OOV word is
replaced by its nearest vocabulary word ONLY when the normalized Levenshtein distance is within
the threshold. Case and punctuation of the original token are preserved.
"""
from htr_sp3 import vectorize
from htr_sp3.corrector import RagCorrector
from htr_sp3.store import InMemoryVectorStore

VOCAB = ["medical", "record", "the", "was", "patient"]


def _corrector(threshold=0.34):
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in VOCAB])
    return RagCorrector(store=store, vocab=set(VOCAB), threshold=threshold)


def test_valid_words_are_left_unchanged():
    text, log = _corrector().correct("the patient record")
    assert text == "the patient record"
    assert log == []


def test_near_oov_word_is_corrected():
    text, log = _corrector().correct("medisal recyrd")
    assert text == "medical record"
    assert {c["from"] for c in log} == {"medisal", "recyrd"}
    assert {c["to"] for c in log} == {"medical", "record"}


def test_far_oov_word_is_left_alone():
    # "xylophone" is nowhere near any vocab word -> distance exceeds threshold -> unchanged.
    text, log = _corrector().correct("xylophone")
    assert text == "xylophone"
    assert log == []


def test_capitalization_is_preserved_on_correction():
    text, _ = _corrector().correct("Medisal")
    assert text == "Medical"


def test_punctuation_and_spacing_are_preserved():
    text, _ = _corrector().correct("medisal, the record.")
    assert text == "medical, the record."


def test_threshold_zero_corrects_nothing():
    text, log = _corrector(threshold=0.0).correct("medisal")
    assert text == "medisal"
    assert log == []


# ---------------------------------------------------------------------------
# Rerank-guard: proves the Levenshtein step matters and cannot be skipped
# ---------------------------------------------------------------------------

# Vocabulary chosen to trigger a cosine-nearest ≠ Levenshtein-winner situation.
# Empirically verified (probe script output):
#
#   store.nearest("handwrit", k=5) returns, in order:
#     "handwritten"  cos_dist=0.2330  lev=0.2727   <- cosine-nearest
#     "handwrite"    cos_dist=0.2372  lev=0.1111   <- Levenshtein winner
#     "handwriting"  cos_dist=0.2984  lev=0.2727
#     "handwrote"    cos_dist=0.4279  lev=0.2222
#     "handwork"     cos_dist=0.5000  lev=0.3750
#
# "handwritten" has more trigram overlap with "handwrit" (shares "handwrit" as a
# prefix giving trigrams ##h, #ha, han, and, nnd, ndw, dwr, wri, rit) but
# "handwrite" differs from the query by only one character ("#e" appended), hence
# its normalized Levenshtein (1/9 ≈ 0.111) beats "handwritten" (3/11 ≈ 0.273).
# The correct/intended word is "handwrite"; "handwritten" would be wrong.
#
# If the Levenshtein rerank were dropped and we simply returned the cosine-nearest
# word, the corrector would output "handwritten" instead of "handwrite" and this
# test would FAIL — that is exactly what this test is designed to catch.
_RERANK_VOCAB = ["handwritten", "handwrite", "handwriting", "handwrote", "handwork"]


def _rerank_corrector(threshold=0.34):
    """Build a corrector using the rerank-guard vocabulary."""
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in _RERANK_VOCAB])
    return RagCorrector(store=store, vocab=set(_RERANK_VOCAB), threshold=threshold)


def test_levenshtein_rerank_overrides_cosine_nearest():
    """Guard the cosine-screen → Levenshtein rerank step.

    The cosine-nearest neighbour of "handwrit" is "handwritten" (highest trigram
    overlap), but "handwrite" has a smaller normalized Levenshtein distance (0.111
    vs 0.273).  The corrector must return "handwrite" — the edit-distance winner —
    not the cosine-nearest "handwritten".  A regression that skips the rerank and
    returns the cosine-nearest directly would output the wrong word and fail here.
    """
    from htr_sp3 import config

    # Sanity check: verify the empirical premise still holds (cosine-nearest is
    # "handwritten", not "handwrite") so this test stays meaningful if internals change.
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in _RERANK_VOCAB])
    hits = store.nearest(vectorize.word_to_vector("handwrit"), k=config.K_NEIGHBORS)
    assert hits[0][0] == "handwritten", (
        "Empirical premise broken: cosine-nearest is no longer 'handwritten'. "
        "Recalibrate the rerank-guard test."
    )

    # The actual behavioural assertion: corrector must pick the Levenshtein winner.
    text, log = _rerank_corrector().correct("handwrit")
    assert text == "handwrite", (
        f"Expected Levenshtein winner 'handwrite' but got {text!r}. "
        "This likely means the rerank step was removed or bypassed."
    )
    assert log[0]["from"] == "handwrit"
    assert log[0]["to"] == "handwrite"


# ---------------------------------------------------------------------------
# possessive-aware gate (Option B): the tokenizer keeps "doll's" as ONE word, so a
# possessive of a perfectly valid noun looks OOV and gets "corrected" (doll's→dollars,
# the headline failure in the investigation report §4). When possessive_aware=True the
# gate also accepts a word whose pre-apostrophe stem is valid, leaving possessives and
# contractions of real words untouched. Default stays False (backward compatible).
# ---------------------------------------------------------------------------

def _possessive_store():
    """Store whose only candidate is 'dollars' — the wrong target for 'doll's'."""
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in ["dollars"]])
    return store


def test_possessive_of_valid_word_left_alone_when_possessive_aware():
    # Gate has the stem "doll"; "doll's" must survive even though "dollars" is in range.
    c = RagCorrector(store=_possessive_store(), vocab={"dollars", "doll"},
                     threshold=0.34, possessive_aware=True)
    text, log = c.correct("doll's")
    assert text == "doll's"
    assert log == []


def test_possessive_is_corrected_when_not_possessive_aware():
    # Default behavior (possessive_aware=False): "doll's" is OOV -> corrected to "dollars".
    c = RagCorrector(store=_possessive_store(), vocab={"dollars", "doll"}, threshold=0.34)
    text, _ = c.correct("doll's")
    assert text == "dollars"


def test_possessive_aware_still_corrects_when_stem_invalid():
    # "xqz's" has no valid stem -> possessive_aware must NOT shield it; normal path applies.
    # ("dollars" is too far from "xqz's", so it ends up unchanged via the threshold, not the gate.)
    c = RagCorrector(store=_possessive_store(), vocab={"dollars", "doll"},
                     threshold=0.34, possessive_aware=True)
    text, _ = c.correct("xqz's")
    assert text == "xqz's"
