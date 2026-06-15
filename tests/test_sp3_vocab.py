"""build_vocabulary extracts the set of unique, normalized words from transcription records.
It is called ONLY on the IAM train split (anti-leakage); the function itself just processes
whatever records it is given.
"""
from htr_sp3 import vocab


def test_lowercases_and_dedupes():
    records = [{"text": "The cat"}, {"text": "the CAT sat"}]
    assert vocab.build_vocabulary(records) == {"the", "cat", "sat"}


def test_strips_surrounding_punctuation():
    records = [{"text": 'He said, "Hello!"'}]
    assert vocab.build_vocabulary(records) == {"he", "said", "hello"}


def test_keeps_intra_word_apostrophes():
    records = [{"text": "don't stop"}]
    assert vocab.build_vocabulary(records) == {"don't", "stop"}


def test_drops_pure_numbers_and_empties():
    records = [{"text": "room 101 ok"}]
    # digits are not words for spelling correction; keep only alphabetic-ish tokens
    assert vocab.build_vocabulary(records) == {"room", "ok"}


# ---------------------------------------------------------------------------
# build_gate_vocabulary: the *validity gate* (Option B) = IAM-train words UNION a
# general English wordlist. This is distinct from build_vocabulary, which stays
# train-only and feeds the candidate STORE (anti-leakage). Widening only the gate
# stops valid English words that happen to be absent from IAM-train (e.g. "sings",
# "stars") from being treated as OOV and "corrected" into the wrong word.
# ---------------------------------------------------------------------------

def test_build_gate_vocabulary_unions_train_and_english():
    records = [{"text": "patient record"}]
    gate = vocab.build_gate_vocabulary(records, {"sings", "stars"})
    assert {"patient", "record", "sings", "stars"} <= gate


def test_build_gate_vocabulary_lowercases_english_words():
    records = [{"text": "the"}]
    gate = vocab.build_gate_vocabulary(records, {"Sings", "STARS"})
    assert {"sings", "stars"} <= gate


def test_build_gate_vocabulary_without_english_equals_train_only():
    records = [{"text": "the cat"}]
    assert vocab.build_gate_vocabulary(records, set()) == vocab.build_vocabulary(records)
