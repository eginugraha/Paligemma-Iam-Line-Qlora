"""Load a general English wordlist for the corrector's validity gate (Option B).

WHY THIS EXISTS
---------------
The SP-3 corrector leaves a word untouched only if it is in the *gate* set. The original gate was
IAM-train words only (~6,851 words), so valid English words that happen not to appear in the train
split (e.g. "sings", "stars", "groups") were treated as OOV and "corrected" into the wrong word —
the over-correction failure diagnosed in docs/sp3-rag-correction-investigation-2026-06-13.md.

This module supplies a large, general English wordlist so the gate can recognise those words.
Crucially, this widens only the GATE; the candidate STORE stays IAM-train-only (anti-leakage),
because a general dictionary is external knowledge, not test/validation labels.

REPRODUCIBILITY
---------------
The wordlist is VENDORED in the repo (data/english_words.txt), not read from a machine-specific
/usr/share/dict/words or downloaded at runtime. So the exact gate is captured in version control
and identical on every machine (laptop, server, RunPod) — same philosophy as the pinned
requirements.txt. data/english_words.txt was generated from the Unix "words" list, lowercased,
filtered to single alphabetic words, and deduped.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Set

# Accept a single word: letters with optional intra-word apostrophes, matching how vocab.py and
# the corrector tokenize. Anything with digits, hyphens, spaces, or punctuation is not a word.
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")

# Default vendored wordlist: <repo-root>/data/english_words.txt (parents[2] == repo root).
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "english_words.txt"


@lru_cache(maxsize=8)
def load_english_words(path: str | None = None) -> frozenset[str]:
    """Return a lowercased, deduped set of English words from a one-word-per-line file.

    Memoized (lru_cache) because the file has ~200k lines and the threshold tuner builds a
    corrector once per candidate threshold — reparsing each time would be wasteful. A frozenset
    is returned so the cached value cannot be mutated by a caller.

    Args:
        path: wordlist file path. Defaults to the vendored data/english_words.txt.

    Returns:
        frozenset of lowercased single words. Lines that are blank or not a single word
        (numbers, hyphenated tokens, etc.) are skipped.
    """
    p = Path(path) if path else _DEFAULT_PATH
    words: Set[str] = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        token = line.strip().lower()
        # fullmatch: the WHOLE line (after strip/lower) must be a single word, else skip it.
        if token and _WORD_RE.fullmatch(token):
            words.add(token)
    return frozenset(words)
