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
