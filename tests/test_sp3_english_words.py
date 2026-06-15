"""load_english_words reads a general English wordlist (one word per line) into a lowercased,
deduped set used as the corrector's *validity gate* (Option B). The wordlist is vendored in the
repo (data/english_words.txt) so the gate is deterministic and reproducible across machines —
it never depends on a machine-specific /usr/share/dict/words or a network download.

Note: load_english_words is memoized (lru_cache), so each test below uses a distinct temp path
to avoid cross-test cache collisions.
"""
from htr_sp3 import english_words


def test_loads_lowercases_and_dedupes(tmp_path):
    f = tmp_path / "dedupe.txt"
    f.write_text("Cat\ncat\nDOG\n")
    assert english_words.load_english_words(str(f)) == frozenset({"cat", "dog"})


def test_skips_non_alpha_and_blank_entries(tmp_path):
    f = tmp_path / "filter.txt"
    f.write_text("hello\n123\nfoo-bar\n\n  spaced  \n")
    words = english_words.load_english_words(str(f))
    assert "hello" in words          # plain word kept
    assert "spaced" in words         # surrounding whitespace stripped
    assert "123" not in words        # pure number dropped
    assert "foo-bar" not in words    # hyphenated token dropped (not a single word)
    assert "" not in words           # blank line dropped


def test_vendored_wordlist_contains_common_words():
    # The default (no-arg) call loads the repo's vendored wordlist. These are exactly the
    # valid words the investigation showed being wrongly "corrected" (see report §4).
    words = english_words.load_english_words()
    for w in ("sings", "stars", "groups", "the", "house"):
        assert w in words, f"expected common word {w!r} in vendored gate wordlist"
