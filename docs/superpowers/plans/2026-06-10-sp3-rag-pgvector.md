# SP-3 RAG / pgvector (M3 & M4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a text-based lexical correction layer (RAG over a vocabulary) implementing scenarios M3 (correct M1 output) and M4 (correct M2/CoT output).

**Architecture:** New `htr_sp3` package. A `RagCorrector` repairs OCR spelling errors word-by-word: char n-gram vectors (feature-hashed to D=512) screen candidate words via cosine in a swappable `VectorStore` (`InMemoryVectorStore` for tests, `PgVectorStore` for production), then a normalized Levenshtein rerank decides, gated by a tuned threshold. The corrector is wired into SP-2's `detect_stream` as an optional post-processor for M3/M4.

**Tech Stack:** Python, pytest, numpy (in-memory cosine), psycopg v3 + PostgreSQL/pgvector (production store), hashlib (deterministic feature hashing). Reuses `htr_sp1.metrics`, `htr_sp1.data`.

**Reference spec:** `docs/superpowers/specs/2026-06-10-sp3-rag-pgvector-design.md`

**Conventions (match SP-1/SP-2):** heavy module/function docstrings and inline comments (thesis must be explainable); heavy imports (numpy/psycopg/datasets) are done lazily inside functions so the package imports instantly on a minimal laptop and unit tests stay fast; tests live in `tests/` and run with plain `pytest`.

---

## File Structure

```
src/htr_sp3/
  __init__.py      # package marker + short purpose docstring
  config.py        # NGRAM_N=3, VECTOR_DIM=512, K_NEIGHBORS=5, DEFAULT_THRESHOLD, HTR_PG_DSN
  vectorize.py     # word_to_vector(): char n-gram -> feature-hashed D=512 -> L2-normalized
  store.py         # VectorStore Protocol + InMemoryVectorStore + PgVectorStore
  vocab.py         # build_vocabulary(records) -> set[str] (lowercased, dedup, filtered)
  corrector.py     # RagCorrector(store, vocab, threshold).correct(text) -> (text, log)
  ingest.py        # build vocab -> vectorize -> store.add_many -> create index
  tune.py          # scan thresholds on validation predictions, pick min-CER T
scripts/
  ingest_sp3.py    # thin CLI to populate the DB
  tune_sp3.py      # thin CLI to run threshold tuning
tests/
  test_sp3_config.py
  test_sp3_vectorize.py
  test_sp3_store.py
  test_sp3_vocab.py
  test_sp3_corrector.py
  test_sp3_orchestrator.py   # extends SP-2 detect_stream with m3/m4
  test_sp3_tune.py
src/htr_sp2/
  config.py        # MODIFY: add M3_STATUS_TAG, M4_STATUS_TAG
  orchestrator.py  # MODIFY: optional corrector param -> emit m3/m4
```

---

## Task 1: Package skeleton + config

**Files:**
- Create: `src/htr_sp3/__init__.py`
- Create: `src/htr_sp3/config.py`
- Test: `tests/test_sp3_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp3_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp3'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp3/__init__.py
"""SP-3: text-based RAG correction over OCR output (scenarios M3 and M4).

The handwriting image is already turned into text by PaliGemma (SP-1/SP-2); SP-3 only repairs
spelling errors in that text against a vocabulary of valid English words from IAM-line. No
images are stored or queried here — this is a lexical corrector, not an image-retrieval system.
"""
```

```python
# src/htr_sp3/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp3_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp3/__init__.py src/htr_sp3/config.py tests/test_sp3_config.py
git commit -m "feat(sp3): package skeleton + RAG config"
```

---

## Task 2: Character n-gram vectorizer

**Files:**
- Create: `src/htr_sp3/vectorize.py`
- Test: `tests/test_sp3_vectorize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp3_vectorize.py
"""word_to_vector turns a word into a fixed-length, deterministic, L2-normalized vector built
from its character trigrams. Similar spellings must produce similar (high-cosine) vectors.
"""
import math

from htr_sp3 import config, vectorize


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized, so dot == cosine


def test_vector_has_fixed_dimension():
    assert len(vectorize.word_to_vector("medical")) == config.VECTOR_DIM


def test_vector_is_deterministic_across_calls():
    # Must not depend on Python's per-process hash randomization (we use hashlib, not hash()).
    assert vectorize.word_to_vector("record") == vectorize.word_to_vector("record")


def test_vector_is_l2_normalized():
    v = vectorize.word_to_vector("handwriting")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_similar_spellings_are_closer_than_dissimilar():
    medical = vectorize.word_to_vector("medical")
    medisal = vectorize.word_to_vector("medisal")   # one-letter OCR error
    zebra = vectorize.word_to_vector("zebra")       # unrelated
    assert _cosine(medical, medisal) > _cosine(medical, zebra)


def test_empty_word_returns_zero_vector():
    v = vectorize.word_to_vector("")
    assert len(v) == config.VECTOR_DIM
    assert all(x == 0.0 for x in v)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_vectorize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp3.vectorize'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp3/vectorize.py
"""Turn a word into a fixed-length character-trigram vector for cosine retrieval.

Approach (classic spelling-similarity vector):
  1. lowercase + pad the word with boundary markers so prefixes/suffixes get their own n-grams,
  2. slice into character n-grams,
  3. feature-hash each n-gram into one of VECTOR_DIM buckets (hashlib -> deterministic across
     processes, unlike Python's salted hash()),
  4. L2-normalize so cosine similarity == dot product.

Why hashing: the raw trigram space is huge and sparse; hashing gives a small dense fixed-size
vector that pgvector can index, while preserving "shares many trigrams => high cosine".
"""
from __future__ import annotations

import hashlib
from typing import List

from . import config

# Boundary marker added around a word so that, e.g., the leading "me" of "medical" becomes a
# distinct trigram ("#me") from a mid-word "me". One marker per side is enough for trigrams.
_PAD = "#"


def _ngrams(word: str, n: int) -> List[str]:
    """Return the character n-grams of *word* after boundary padding.

    Args:
        word: already-lowercased word.
        n: n-gram size (config.NGRAM_N).

    Returns:
        List of n-character substrings; empty if the word is empty.
    """
    if not word:
        return []
    padded = _PAD * (n - 1) + word + _PAD * (n - 1)
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def _bucket(ngram: str) -> int:
    """Hash an n-gram to a stable bucket index in [0, VECTOR_DIM).

    Uses md5 (via hashlib) rather than the builtin hash() because hash() is randomized per
    process (PYTHONHASHSEED), which would make vectors differ between ingest and query runs.
    """
    digest = hashlib.md5(ngram.encode("utf-8")).hexdigest()
    return int(digest, 16) % config.VECTOR_DIM


def word_to_vector(word: str) -> List[float]:
    """Vectorize a single word into a fixed-length, L2-normalized list of floats.

    Args:
        word: any string; case is ignored.

    Returns:
        A list of length config.VECTOR_DIM. All zeros for an empty word.
    """
    vec = [0.0] * config.VECTOR_DIM
    for ng in _ngrams(word.lower(), config.NGRAM_N):
        vec[_bucket(ng)] += 1.0

    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec  # empty word -> zero vector (callers treat this as "no signal")
    return [x / norm for x in vec]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp3_vectorize.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp3/vectorize.py tests/test_sp3_vectorize.py
git commit -m "feat(sp3): char n-gram word vectorizer (feature-hashed, L2-normalized)"
```

---

## Task 3: VectorStore protocol + InMemoryVectorStore

**Files:**
- Create: `src/htr_sp3/store.py`
- Test: `tests/test_sp3_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp3_store.py
"""The store holds (word, vector) rows and returns nearest words by cosine DISTANCE
(1 - cosine similarity), ascending. InMemoryVectorStore is the test/dev implementation;
it satisfies the same VectorStore Protocol as the production PgVectorStore.
"""
from htr_sp3 import vectorize
from htr_sp3.store import InMemoryVectorStore


def _populate():
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in ["medical", "record", "zebra"]])
    return store


def test_nearest_returns_closest_word_first():
    store = _populate()
    query = vectorize.word_to_vector("medisal")  # typo of "medical"
    results = store.nearest(query, k=3)
    assert results[0][0] == "medical"


def test_nearest_respects_k():
    store = _populate()
    query = vectorize.word_to_vector("medisal")
    assert len(store.nearest(query, k=2)) == 2


def test_nearest_results_sorted_by_ascending_distance():
    store = _populate()
    query = vectorize.word_to_vector("medisal")
    distances = [dist for _, dist in store.nearest(query, k=3)]
    assert distances == sorted(distances)


def test_empty_store_returns_empty():
    assert InMemoryVectorStore().nearest(vectorize.word_to_vector("x"), k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp3.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp3/store.py
"""Vector storage for the vocabulary, behind a swappable interface.

`VectorStore` is a Protocol (structural type) so the corrector depends on the behaviour, not a
concrete class — exactly the InferenceEngine/FakeEngine pattern from SP-2. Two implementations:

  - InMemoryVectorStore : pure-numpy cosine; used by the whole test suite and small local runs,
                          so NO PostgreSQL is needed to develop or test SP-3.
  - PgVectorStore       : PostgreSQL + pgvector for production (added in a later task).
"""
from __future__ import annotations

from typing import List, Protocol, Tuple

# A retrieval hit: (word, cosine_distance) where distance = 1 - cosine_similarity (0 == identical).
Hit = Tuple[str, float]
# An ingest row: (word, vector).
Row = Tuple[str, List[float]]


class VectorStore(Protocol):
    """Behaviour every store must provide."""

    def add_many(self, rows: List[Row]) -> None:
        """Insert (word, vector) rows."""
        ...

    def nearest(self, vector: List[float], k: int) -> List[Hit]:
        """Return the k nearest words to *vector* as (word, distance), ascending by distance."""
        ...


class InMemoryVectorStore:
    """Reference VectorStore using numpy. Vectors are assumed L2-normalized (vectorize does this),
    so cosine similarity is a plain dot product and distance is 1 - dot.
    """

    def __init__(self) -> None:
        self._words: List[str] = []
        self._matrix = None  # lazily built numpy array of shape (N, dim)

    def add_many(self, rows: List[Row]) -> None:
        import numpy as np

        for word, vec in rows:
            self._words.append(word)
        new = np.array([vec for _, vec in rows], dtype="float32")
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    def nearest(self, vector: List[float], k: int) -> List[Hit]:
        import numpy as np

        if self._matrix is None or len(self._words) == 0:
            return []
        q = np.array(vector, dtype="float32")
        sims = self._matrix @ q            # dot product == cosine (vectors are normalized)
        distances = 1.0 - sims
        order = np.argsort(distances)[:k]  # ascending distance
        return [(self._words[i], float(distances[i])) for i in order]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp3_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp3/store.py tests/test_sp3_store.py
git commit -m "feat(sp3): VectorStore protocol + InMemoryVectorStore"
```

---

## Task 4: Vocabulary builder

**Files:**
- Create: `src/htr_sp3/vocab.py`
- Test: `tests/test_sp3_vocab.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp3_vocab.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_vocab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp3.vocab'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp3/vocab.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp3_vocab.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp3/vocab.py tests/test_sp3_vocab.py
git commit -m "feat(sp3): vocabulary builder (train-only, normalized, deduped)"
```

---

## Task 5: RagCorrector

**Files:**
- Create: `src/htr_sp3/corrector.py`
- Test: `tests/test_sp3_corrector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp3_corrector.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_corrector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp3.corrector'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp3/corrector.py
"""The RAG corrector: repair OCR spelling errors against a vocabulary.

Pipeline per word (see the SP-3 design doc):
  1. tokenize the text into word / non-word chunks so the original spacing & punctuation can be
     rebuilt verbatim;
  2. a word already in the vocabulary is kept as-is (idempotent on correct words);
  3. an out-of-vocabulary word is vectorized and the store returns the cosine-nearest candidates;
  4. those candidates are reranked by NORMALIZED LEVENSHTEIN distance (cosine screens, edit
     distance decides — it is the better judge of "same word, mistyped");
  5. the best candidate replaces the word ONLY if its distance <= threshold; otherwise the
     original word is kept (protects proper nouns and genuine OOV);
  6. the replacement inherits the original token's capitalization.

The corrector depends only on a `VectorStore` (for retrieval) and a `vocab` set (for the
exact-match gate), so it is fully testable with InMemoryVectorStore and no database.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from . import config, vectorize
from .store import VectorStore

# Split text into alternating word / non-word chunks, keeping BOTH so we can rejoin losslessly.
# A word allows intra-word apostrophes ("don't"); everything else (spaces, punctuation, digits)
# is a non-word chunk that passes through untouched.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^A-Za-z]+")


def _normalized_levenshtein(a: str, b: str) -> float:
    """Levenshtein edit distance between *a* and *b*, divided by the longer length (0..1)."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _match_case(original: str, replacement: str) -> str:
    """Apply *original*'s capitalization pattern to the lowercase *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


class RagCorrector:
    """Correct OCR text using vocabulary retrieval + edit-distance rerank."""

    def __init__(self, store: VectorStore, vocab: Set[str], threshold: float = config.DEFAULT_THRESHOLD) -> None:
        """
        Args:
            store:     populated VectorStore (in-memory for tests, pgvector in production).
            vocab:     set of valid lowercased words for the exact-match gate.
            threshold: max normalized Levenshtein distance to accept a correction (0..1).
        """
        self._store = store
        self._vocab = vocab
        self._threshold = threshold

    def correct(self, text: str) -> Tuple[str, List[Dict[str, object]]]:
        """Return (corrected_text, log) where log lists {from, to, distance} per replacement."""
        out: List[str] = []
        log: List[Dict[str, object]] = []

        for chunk in _TOKEN_RE.findall(text):
            # Non-word chunk (spaces/punctuation/digits) -> pass through unchanged.
            if not chunk[:1].isalpha():
                out.append(chunk)
                continue

            lower = chunk.lower()
            # Already a valid word -> keep it (no correction).
            if lower in self._vocab:
                out.append(chunk)
                continue

            best = self._best_candidate(lower)
            if best is not None and best[1] <= self._threshold:
                word, distance = best
                out.append(_match_case(chunk, word))
                log.append({"from": lower, "to": word, "distance": round(distance, 4)})
            else:
                out.append(chunk)  # too far / no candidate -> leave original

        return "".join(out), log

    def _best_candidate(self, word: str) -> Tuple[str, float] | None:
        """Return the (candidate, normalized_levenshtein) with the smallest edit distance among
        the cosine-nearest neighbours, or None if the store is empty.
        """
        hits = self._store.nearest(vectorize.word_to_vector(word), k=config.K_NEIGHBORS)
        if not hits:
            return None
        ranked = [(cand, _normalized_levenshtein(word, cand)) for cand, _cos in hits]
        ranked.sort(key=lambda pair: pair[1])
        return ranked[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp3_corrector.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp3/corrector.py tests/test_sp3_corrector.py
git commit -m "feat(sp3): RagCorrector (cosine screen + Levenshtein gate, case/punct preserved)"
```

---

## Task 6: Wire M3/M4 into the orchestrator

**Files:**
- Modify: `src/htr_sp2/config.py` (add status tags)
- Modify: `src/htr_sp2/orchestrator.py` (optional corrector -> emit m3/m4)
- Test: `tests/test_sp3_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp3_orchestrator.py
"""detect_stream gains an optional `corrector`. When present, after M1/M2 it emits m3 (correct
M1 text) and m4 (correct M2 text). When absent, behaviour is unchanged (backward-compatible).
A correction error isolates to its own error event; the stream still finishes with `done`.
"""
import json

from htr_sp2.engine import EngineError
from htr_sp2.orchestrator import detect_stream


class FakeEngine:
    """Returns canned raw outputs per prompt so we control M1/M2 text."""

    def __init__(self, m1="medisal", m2="reasoning... Final: recyrd"):
        self._m1, self._m2 = m1, m2

    def run(self, image, prompt, max_new_tokens):
        # M2's CoT prompt is longer; distinguish by which prompt arrives.
        from htr_sp2 import config
        return self._m2 if prompt == config.M2_PROMPT else self._m1


class FakeCorrector:
    def correct(self, text):
        mapping = {"medisal": "medical", "recyrd": "record"}
        fixed = mapping.get(text.strip(), text)
        log = [] if fixed == text else [{"from": text, "to": fixed, "distance": 0.1}]
        return fixed, log


def _events(gen):
    return [json.loads(line) for line in gen]


def test_without_corrector_only_m1_m2():
    events = _events(detect_stream(FakeEngine(), image=None, filename="x.png", ground_truth=None))
    models = [e["model"] for e in events if e.get("event") == "result"]
    assert models == ["m1", "m2"]


def test_with_corrector_emits_m3_and_m4():
    events = _events(detect_stream(
        FakeEngine(), image=None, filename="x.png", ground_truth=None, corrector=FakeCorrector()
    ))
    results = {e["model"]: e for e in events if e.get("event") == "result"}
    assert set(results) == {"m1", "m2", "m3", "m4"}
    assert results["m3"]["text"] == "medical"   # corrected M1
    assert results["m4"]["text"] == "record"    # corrected M2


def test_m4_skipped_when_m2_fails():
    class M2FailEngine(FakeEngine):
        def run(self, image, prompt, max_new_tokens):
            from htr_sp2 import config
            if prompt == config.M2_PROMPT:
                raise EngineError("boom")
            return self._m1

    events = _events(detect_stream(
        M2FailEngine(), image=None, filename="x.png", ground_truth=None, corrector=FakeCorrector()
    ))
    errors = {e["model"] for e in events if e.get("event") == "error"}
    assert "m2" in errors and "m4" in errors
    assert events[-1]["event"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_orchestrator.py -v`
Expected: FAIL — `detect_stream() got an unexpected keyword argument 'corrector'`

- [ ] **Step 3a: Add status tags to SP-2 config**

Append to `src/htr_sp2/config.py`:

```python
# M3 (RAG correction of M1) and M4 (RAG correction of M2/CoT) status badges shown by the
# frontend table. Kept here with the M1/M2 tags so all scenario labels live in one file.
M3_STATUS_TAG = "Corrected"
M4_STATUS_TAG = "Optimal"
```

- [ ] **Step 3b: Add the corrector param + m3/m4 emission to the orchestrator**

In `src/htr_sp2/orchestrator.py`, change the `detect_stream` signature to add the optional
parameter (keep the existing parameters and docstring; add the new arg + a line documenting it):

```python
def detect_stream(
    engine: InferenceEngine,
    image: Image.Image,
    filename: str,
    ground_truth: str | None,
    corrector=None,  # optional htr_sp3.corrector.RagCorrector; when set, emit m3 & m4
) -> Iterator[str]:
```

Then, capture the M1/M2 corrected-source text inside the existing scenario loop. Replace the
`if spec.model == "m1": ... else: ...` block (the post-processing block) with a version that
remembers each scenario's final text so M3/M4 can reuse it:

```python
        # --- 2a. Post-process the raw output depending on scenario ------------
        if spec.model == "m1":
            text = raw.strip()
            log = M1_LOG
            m1_text = text          # remember for M3 (RAG correction of the baseline)
        else:
            text, log = parse_cot(raw)
            m2_text = text          # remember for M4 (RAG correction of the CoT answer)
```

Initialize `m1_text = None` and `m2_text = None` just before the `for spec in _SPECS:` loop so
they exist even if a scenario fails.

Finally, after the existing scenario loop and BEFORE the `done` event, add the RAG scenarios:

```python
    # --- 2.5 RAG correction scenarios (only when a corrector is supplied) ------
    # M3 corrects M1's text; M4 corrects M2's text. Each is isolated: a failure emits an error
    # event for that scenario and the stream continues. M4 is skipped if M2 produced no text.
    if corrector is not None:
        _rag_sources = [
            ("m3", m1_text, config.M3_STATUS_TAG, "m1"),
            ("m4", m2_text, config.M4_STATUS_TAG, "m2"),
        ]
        for model, source_text, status_tag, depends_on in _rag_sources:
            if source_text is None:
                yield _line(schemas.error_event(model, f"depends on {depends_on} which failed"))
                continue
            try:
                start = time.perf_counter()
                corrected, corrections = corrector.correct(source_text)
                latency = round(time.perf_counter() - start, 3)
            except Exception as exc:  # corrector/store failure isolates to this scenario
                yield _line(schemas.error_event(model, str(exc)))
                continue

            if ground_truth is not None:
                cer_value = round(cer_metric(ground_truth, corrected), 2)
                wer_value = round(wer_metric(ground_truth, corrected), 2)
            else:
                cer_value = wer_value = None

            # Human-readable correction log for the frontend tooltip / thesis appendix.
            log = "RAG: " + (
                ", ".join(f"{c['from']}→{c['to']} ({c['distance']})" for c in corrections)
                if corrections else "no corrections"
            )
            yield _line(schemas.result_event(
                model=model, text=corrected, cer=cer_value, wer=wer_value,
                latency_seconds=latency, log=log, status_tag=status_tag,
            ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sp3_orchestrator.py tests/test_sp2_orchestrator.py -v`
Expected: PASS (new m3/m4 tests pass; all existing SP-2 orchestrator tests still pass)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `python -m pytest -q`
Expected: PASS (all SP-1, SP-2, SP-3 tests green)

- [ ] **Step 6: Commit**

```bash
git add src/htr_sp2/config.py src/htr_sp2/orchestrator.py tests/test_sp3_orchestrator.py
git commit -m "feat(sp3): emit M3/M4 RAG scenarios from detect_stream (optional corrector)"
```

---

## Task 7: PgVectorStore (production store)

**Files:**
- Modify: `src/htr_sp3/store.py` (add `PgVectorStore`)
- Modify: `requirements.txt` (add `psycopg[binary]` and note pgvector)
- Test: `tests/test_sp3_store.py` (add a DB test that SKIPS without `HTR_PG_DSN`)

> **Note:** `PgVectorStore` talks to a real database, so it is verified manually (Task 8) and via
> one opt-in test that is skipped in CI. No TDD red/green for the network path — the logic is a
> thin wrapper over SQL. Keep the in-memory store as the behaviour reference.

- [ ] **Step 1: Add the opt-in DB test**

```python
# append to tests/test_sp3_store.py
import os

import pytest

from htr_sp3.config import VOCAB_TABLE  # noqa: E402


@pytest.mark.skipif(not os.environ.get("HTR_PG_DSN"), reason="no Postgres DSN; pgvector test skipped")
def test_pgvector_roundtrip():
    from htr_sp3.store import PgVectorStore

    store = PgVectorStore()
    store.create_schema()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in ["medical", "record"]])
    store.create_index()
    results = store.nearest(vectorize.word_to_vector("medisal"), k=1)
    assert results[0][0] == "medical"
```

- [ ] **Step 2: Implement PgVectorStore**

Append to `src/htr_sp3/store.py`:

```python
class PgVectorStore:
    """Production VectorStore backed by PostgreSQL + the pgvector extension.

    Cosine distance uses pgvector's `<=>` operator. Vectors are stored as the pgvector `vector`
    type; we pass them as the literal string "[v1,v2,...]" which the type accepts. psycopg v3 is
    imported lazily so the rest of SP-3 (and the test suite) never requires a DB driver.
    """

    def __init__(self, dsn: str | None = None) -> None:
        from . import config
        self._dsn = dsn or config.PG_DSN
        self._table = config.VOCAB_TABLE
        self._dim = config.VECTOR_DIM

    def _connect(self):
        import psycopg
        return psycopg.connect(self._dsn)

    @staticmethod
    def _to_literal(vector: List[float]) -> str:
        # pgvector accepts a text literal like "[0.1,0.2,...]".
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    def create_schema(self) -> None:
        """Create the pgvector extension + vocab table (idempotent). Truncates existing rows."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} "
                f"(word TEXT PRIMARY KEY, vec vector({self._dim}))"
            )
            cur.execute(f"TRUNCATE {self._table}")
            conn.commit()

    def add_many(self, rows: List[Row]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._table} (word, vec) VALUES (%s, %s) "
                f"ON CONFLICT (word) DO UPDATE SET vec = EXCLUDED.vec",
                [(word, self._to_literal(vec)) for word, vec in rows],
            )
            conn.commit()

    def create_index(self) -> None:
        """Build the HNSW cosine index after bulk insert (faster than inserting into an index)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_vec_hnsw "
                f"ON {self._table} USING hnsw (vec vector_cosine_ops)"
            )
            conn.commit()

    def nearest(self, vector: List[float], k: int) -> List[Hit]:
        lit = self._to_literal(vector)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT word, vec <=> %s AS distance FROM {self._table} "
                f"ORDER BY vec <=> %s LIMIT %s",
                (lit, lit, k),
            )
            return [(word, float(dist)) for word, dist in cur.fetchall()]
```

- [ ] **Step 3: Update requirements**

Add to `requirements.txt`:

```
psycopg[binary]>=3.1   # PostgreSQL driver for the pgvector store (SP-3)
# Requires a PostgreSQL server with the `pgvector` extension available (CREATE EXTENSION vector).
```

- [ ] **Step 4: Verify the suite still passes (DB test skips)**

Run: `python -m pytest tests/test_sp3_store.py -v`
Expected: PASS for in-memory tests; `test_pgvector_roundtrip` SKIPPED (no `HTR_PG_DSN`).

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp3/store.py tests/test_sp3_store.py requirements.txt
git commit -m "feat(sp3): PgVectorStore (pgvector cosine) + opt-in DB test"
```

---

## Task 8: Ingestion module + CLI

**Files:**
- Create: `src/htr_sp3/ingest.py`
- Create: `scripts/ingest_sp3.py`
- Test: `tests/test_sp3_ingest.py`

- [ ] **Step 1: Write the failing test (logic only, store injected)**

```python
# tests/test_sp3_ingest.py
"""ingest_vocabulary builds the vocab, vectorizes each word, and loads it into ANY store. We
inject an InMemoryVectorStore so the test needs no database, and assert the words are queryable.
"""
from htr_sp3 import ingest, vectorize
from htr_sp3.store import InMemoryVectorStore


def test_ingest_populates_store_from_records():
    records = [{"text": "the medical record"}, {"text": "the patient"}]
    store = InMemoryVectorStore()

    count = ingest.ingest_vocabulary(records, store)

    assert count == 4  # the, medical, record, patient
    nearest = store.nearest(vectorize.word_to_vector("medisal"), k=1)
    assert nearest[0][0] == "medical"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp3.ingest'`

- [ ] **Step 3: Implement ingest + CLI**

```python
# src/htr_sp3/ingest.py
"""Populate a VectorStore with the IAM vocabulary.

`ingest_vocabulary` is store-agnostic (takes any VectorStore) so it is unit-tested with the
in-memory store and reused by the CLI with PgVectorStore. The CLI wiring (load IAM, build the
pg store, create schema/index) lives in scripts/ingest_sp3.py.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

from . import vectorize, vocab
from .store import VectorStore


def ingest_vocabulary(records: Iterable[Dict[str, Any]], store: VectorStore) -> int:
    """Build the vocabulary from *records* and load (word, vector) rows into *store*.

    Args:
        records: IAM TRAIN split records ({"text": ...}). Train only — see vocab.build_vocabulary.
        store:   a VectorStore to populate.

    Returns:
        Number of unique words ingested.
    """
    words = sorted(vocab.build_vocabulary(records))  # sorted -> deterministic ingest order
    rows = [(w, vectorize.word_to_vector(w)) for w in words]
    store.add_many(rows)
    return len(rows)
```

```python
# scripts/ingest_sp3.py
#!/usr/bin/env python
"""SP-3 ingestion CLI: build the IAM-train vocabulary and load it into PostgreSQL/pgvector.

Usage:
    export HTR_PG_DSN="postgresql://user:pass@localhost:5432/htr"
    python scripts/ingest_sp3.py
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htr_sp1 import data  # noqa: E402
from htr_sp3 import ingest  # noqa: E402
from htr_sp3.store import PgVectorStore  # noqa: E402


def main() -> None:
    print("[SP-3 ingest] loading IAM train split...")
    train = data.load_iam_splits()["train"]

    store = PgVectorStore()
    print("[SP-3 ingest] creating schema (truncates existing rows)...")
    store.create_schema()

    print("[SP-3 ingest] building + loading vocabulary...")
    count = ingest.ingest_vocabulary(train, store)

    print(f"[SP-3 ingest] building HNSW index over {count} words...")
    store.create_index()
    print(f"[SP-3 ingest] done. {count} words ingested.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp3_ingest.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp3/ingest.py scripts/ingest_sp3.py tests/test_sp3_ingest.py
git commit -m "feat(sp3): vocabulary ingestion module + CLI"
```

---

## Task 9: Threshold tuning on validation

**Files:**
- Create: `src/htr_sp3/tune.py`
- Create: `scripts/tune_sp3.py`
- Test: `tests/test_sp3_tune.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sp3_tune.py
"""tune_threshold scans candidate thresholds and returns the one with the lowest mean CER on the
provided (prediction, ground_truth) pairs, plus the per-threshold curve. It is store/model
agnostic: we inject an in-memory corrector factory so no DB or model is needed.
"""
from htr_sp3 import tune, vectorize
from htr_sp3.corrector import RagCorrector
from htr_sp3.store import InMemoryVectorStore

VOCAB = ["medical", "record", "the", "patient", "was"]


def _make_corrector(threshold):
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in VOCAB])
    return RagCorrector(store=store, vocab=set(VOCAB), threshold=threshold)


def test_tune_picks_threshold_that_minimizes_cer():
    # Predictions have OCR errors that correction fixes; ground truth is the clean text.
    pairs = [("medisal", "medical"), ("recyrd", "record"), ("the", "the")]
    result = tune.tune_threshold(pairs, _make_corrector, thresholds=[0.0, 0.34])

    # 0.0 corrects nothing (high CER); 0.34 fixes the typos (CER 0) -> best is 0.34.
    assert result["best_threshold"] == 0.34
    assert result["best_cer"] < result["curve"][0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sp3_tune.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp3.tune'`

- [ ] **Step 3: Implement tune + CLI**

```python
# src/htr_sp3/tune.py
"""Pick the correction threshold that minimizes validation CER.

Thesis integrity: tune on VALIDATION predictions, never test. `tune_threshold` is pure logic —
it takes (prediction, ground_truth) pairs and a factory that builds a corrector for a given
threshold — so it is unit-testable with an in-memory corrector and no model/DB. The CLI
(scripts/tune_sp3.py) supplies real M1 predictions on the validation split.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from htr_sp1.metrics import cer as cer_metric

# A corrector factory: threshold -> object with .correct(text) -> (text, log).
CorrectorFactory = Callable[[float], object]


def tune_threshold(
    pairs: List[Tuple[str, str]],
    make_corrector: CorrectorFactory,
    thresholds: List[float],
) -> Dict[str, object]:
    """Return the best threshold and the full CER-vs-threshold curve.

    Args:
        pairs:          list of (prediction, ground_truth) on the validation split.
        make_corrector: builds a corrector for a given threshold.
        thresholds:     candidate thresholds to scan (e.g. [0.10, 0.15, ... 0.50]).

    Returns:
        {"best_threshold": float, "best_cer": float, "curve": {threshold: mean_cer}}.
    """
    curve: Dict[float, float] = {}
    for t in thresholds:
        corrector = make_corrector(t)
        total = 0.0
        for prediction, truth in pairs:
            corrected, _log = corrector.correct(prediction)
            total += cer_metric(truth, corrected)
        curve[t] = total / len(pairs) if pairs else 0.0

    best_threshold = min(curve, key=curve.get)
    return {"best_threshold": best_threshold, "best_cer": curve[best_threshold], "curve": curve}
```

```python
# scripts/tune_sp3.py
#!/usr/bin/env python
"""SP-3 threshold tuning CLI: scan thresholds on validation and write the best one.

Needs (a) a populated pgvector store (run scripts/ingest_sp3.py first) and (b) M1 predictions on
the IAM validation split as a JSON list of {"prediction": ..., "ground_truth": ...}. Produce that
file from the SP-1 eval (scripts/eval_sp1.py writes per_sample rows you can adapt) or any M1 run.

Usage:
    export HTR_PG_DSN="postgresql://localhost:5432/htr"
    python scripts/tune_sp3.py --pairs val_m1_predictions.json --out tune_sp3.json
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htr_sp1 import data  # noqa: E402
from htr_sp3 import tune, vocab  # noqa: E402
from htr_sp3.corrector import RagCorrector  # noqa: E402
from htr_sp3.store import PgVectorStore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Tune the SP-3 correction threshold on validation.")
    p.add_argument("--pairs", required=True, help="JSON list of {prediction, ground_truth}.")
    p.add_argument("--out", default="tune_sp3.json", help="Where to write the tuning result.")
    args = p.parse_args()

    pairs_raw = json.loads(Path(args.pairs).read_text())
    pairs = [(r["prediction"], r["ground_truth"]) for r in pairs_raw]

    # Vocab set (for the exact-match gate) from IAM train — same source as ingest.
    vocab_set = vocab.build_vocabulary(data.load_iam_splits()["train"])
    store = PgVectorStore()  # already populated by ingest_sp3.py

    def make_corrector(threshold: float) -> RagCorrector:
        return RagCorrector(store=store, vocab=vocab_set, threshold=threshold)

    grid = [round(0.10 + 0.05 * i, 2) for i in range(9)]  # 0.10 .. 0.50
    grid = [0.0] + grid                                   # include "no correction" baseline
    result = tune.tune_threshold(pairs, make_corrector, grid)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[SP-3 tune] best_threshold={result['best_threshold']} "
          f"best_cer={result['best_cer']:.2f} (baseline T=0.0 CER={result['curve'][0.0]:.2f})")
    print(f"[SP-3 tune] wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sp3_tune.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (all SP-1/SP-2/SP-3 tests green; pgvector DB test skipped)

- [ ] **Step 6: Commit**

```bash
git add src/htr_sp3/tune.py scripts/tune_sp3.py tests/test_sp3_tune.py
git commit -m "feat(sp3): validation threshold tuning (min-CER) module + CLI"
```

---

## Post-implementation (manual, after tomorrow's PaliGemma re-train)

These need the trained model and/or a live database, so they run outside the TDD loop:

1. **Start PostgreSQL + pgvector**, set `HTR_PG_DSN`, run `python scripts/ingest_sp3.py`.
2. **Produce M1 validation predictions** (adapt `scripts/eval_sp1.py` to dump
   `{prediction, ground_truth}` for the validation split), then run `python scripts/tune_sp3.py`.
3. **Set the tuned threshold** as `config.DEFAULT_THRESHOLD` (or pass it through at runtime).
4. **Run the opt-in DB test** once: `HTR_PG_DSN=... python -m pytest tests/test_sp3_store.py -k pgvector`.
5. **Update PRD/methodology** to record the RAG design (char n-gram + Levenshtein, train-only
   vocab, validation-tuned threshold) for Chapter 4.

---

## Self-Review

- **Spec coverage:** package/components (Tasks 1-5), pgvector schema+store (Task 7), ingestion
  (Task 8), correction flow (Task 5), orchestrator M3/M4 + status tags + failure isolation +
  M4-depends-on-M2 (Task 6), tuning + no-correction baseline (Task 9), testing-without-Postgres
  (in-memory store throughout). All spec sections map to a task.
- **Placeholders:** none — every step ships runnable code and exact commands.
- **Type consistency:** `VectorStore.add_many(rows)` / `nearest(vector, k) -> List[Hit]`,
  `RagCorrector(store, vocab, threshold).correct(text) -> (str, list)`,
  `ingest_vocabulary(records, store) -> int`, `tune_threshold(pairs, make_corrector, thresholds)
  -> {best_threshold, best_cer, curve}` are used consistently across tasks and CLIs.
```
