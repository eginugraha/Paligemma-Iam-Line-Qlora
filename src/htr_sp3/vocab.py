"""Build the correction vocabulary from transcription records.

CRITICAL (thesis integrity): call this on the IAM TRAIN split only. Building the vocabulary
from validation/test transcriptions would leak the answers into the corrector and inflate the
reported CER/WER gains. The function is split out (and unit-tested) so this rule is explicit.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Set

# A "word" is a run of letters with optional intra-word apostrophes (so "don't" stays whole).
# Pure numbers and punctuation are excluded — they are not spelling-correction targets.
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


def build_vocabulary(records: Iterable[Dict[str, Any]]) -> Set[str]:
    """Return the set of unique, lowercased words across all records' "text" field.

    Args:
        records: iterable of {"text": <transcription>} (e.g. the IAM train split).

    Returns:
        Set of normalized vocabulary words.
    """
    vocab: Set[str] = set()
    for record in records:
        text = record["text"].lower()
        vocab.update(_WORD_RE.findall(text))
    return vocab


def build_gate_vocabulary(records: Iterable[Dict[str, Any]], english_words: Iterable[str]) -> Set[str]:
    """Return the corrector's *validity gate*: IAM-train words UNION a general English wordlist.

    WHY THIS IS SEPARATE FROM build_vocabulary (thesis integrity):
    - build_vocabulary (train-only) feeds the candidate STORE — keeping it train-only is the
      anti-leakage guarantee (test/validation answers never become correction candidates).
    - build_gate_vocabulary widens only the *gate* — the set of words the corrector treats as
      already-valid and leaves untouched. A general English wordlist is external knowledge, NOT
      test labels, so adding it does not leak. It fixes the over-correction failure where valid
      English words absent from IAM-train (e.g. "sings", "stars") were treated as OOV and changed.

    Args:
        records:       IAM-train records ({"text": ...}); same source as build_vocabulary.
        english_words: a general English wordlist (any iterable of strings); lowercased here.

    Returns:
        The union set used as the exact-match gate.
    """
    return build_vocabulary(records) | {w.lower() for w in english_words}
