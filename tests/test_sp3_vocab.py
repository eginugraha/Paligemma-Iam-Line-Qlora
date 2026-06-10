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
