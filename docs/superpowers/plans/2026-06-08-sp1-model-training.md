# SP-1 Model Training & Packaging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a well-commented, reproducible package + Colab notebook that fine-tunes PaliGemma-3B-PT-448 (QLoRA) on IAM-line for handwriting transcription, evaluates CER/WER, and exports a validated model to the Hugging Face Hub.

**Architecture:** Pure, unit-testable Python functions in `src/htr_sp1/` (config, data, metrics, model, inference, train, evaluate, export). Heavy GPU work (model load, training) is parameterized so tests inject fakes and run on a CPU laptop with no downloads; a thin `notebooks/sp1_train.ipynb` orchestrates the real run in Colab on a T4.

**Tech Stack:** Python 3.10+, PyTorch, `transformers` (PaliGemma), `peft` (LoRA), `bitsandbytes` (4-bit/QLoRA), `datasets` (Teklia/IAM-line), `jiwer` (CER/WER), `huggingface_hub`, `pytest`.

> **Documentation requirement (applies to EVERY task):** The user must be able to read and explain every line for their thesis defense. Every function gets a docstring stating purpose, args, returns, and *why*. Every non-obvious line gets an inline comment explaining the reasoning, not just the mechanics. Do not strip these comments to "clean up."

---

## File Structure

```
src/htr_sp1/
  __init__.py
  config.py        # All hyperparameters, paths, seed, prompt constant — single source of truth
  metrics.py       # CER / WER via jiwer (pure, fully testable)
  data.py          # Load IAM-line, build prompt, turn a record into a processor example
  model.py         # Build 4-bit base + attach LoRA (assembly logic testable via injection)
  inference.py     # generate_transcription(model, processor, image) — THE interface SP-2 consumes
  train.py         # Assemble TrainingArguments + Trainer, checkpoint/resume (assembly testable)
  evaluate.py      # Run inference over a split, aggregate CER/WER (testable via injected infer fn)
  export.py        # Merge LoRA, push adapter + merged to HF Hub (path/id logic testable, push mocked)
notebooks/
  sp1_train.ipynb  # Thin orchestration for Colab (manual-run gate, not unit tested)
tests/
  conftest.py      # Shared fakes: FakeProcessor, FakeModel, tiny synthetic records
  test_config.py
  test_metrics.py
  test_data.py
  test_model.py
  test_inference.py
  test_train.py
  test_evaluate.py
  test_export.py
requirements.txt   # Pinned versions for reproducibility
README-sp1.md      # How to run locally (tests) and in Colab; the documented inference interface
pytest.ini
```

**Decomposition rationale:** one responsibility per file. `config.py` is the only place numbers/paths live (DRY). `inference.py` is deliberately separate because it is the public contract SP-2 imports — it must stay small and stable. Files that change together (e.g. data + prompt) live together.

---

## Task 0: Project scaffold + pinned dependencies

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `src/htr_sp1/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt` with pinned versions**

```text
# Pinned for reproducibility — these versions are known to work together for
# PaliGemma QLoRA on a Colab T4. Do NOT bump without re-validating training.
torch==2.3.1
transformers==4.42.4
peft==0.11.1
bitsandbytes==0.43.1
accelerate==0.31.0
datasets==2.20.0
jiwer==3.0.4
huggingface_hub==0.23.4
pillow==10.4.0
# Dev / test only (CPU laptop):
pytest==8.2.2
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
# Look for tests in tests/, treat src/ as importable (we run `pip install -e .`-free
# by adding src to the path via the conftest at repo root).
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Create empty package markers**

`src/htr_sp1/__init__.py`:
```python
"""SP-1: PaliGemma QLoRA fine-tuning for line-level handwriting transcription (IAM-line).

This package holds small, single-responsibility, heavily-documented modules so that
every step of the thesis pipeline can be read, tested, and explained independently.
"""
```

`tests/__init__.py`:
```python
# Marker so pytest can import the tests package.
```

- [ ] **Step 4: Make `src/` importable in tests — create repo-root `conftest.py`**

`conftest.py` (at repo root):
```python
"""Pytest bootstrap: put `src/` on sys.path so `import htr_sp1...` works without
installing the package. Keeps local test runs friction-free for the thesis team.
"""
import sys
from pathlib import Path

# src/ sits next to this file; prepend it so the htr_sp1 package is importable.
SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
```

- [ ] **Step 5: Verify pytest collects nothing yet (sanity)**

Run: `python -m pytest`
Expected: `no tests ran` (exit code 5) — confirms config is valid and discovery works.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini conftest.py src/htr_sp1/__init__.py tests/__init__.py
git commit -m "chore: scaffold SP-1 package, pinned deps, pytest setup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: Config module (single source of truth)

**Files:**
- Create: `src/htr_sp1/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
"""Config is the only place hyperparameters/paths live. These tests pin the
contract the rest of the pipeline relies on, and prove `set_seed` is deterministic.
"""
import random
from htr_sp1 import config


def test_core_constants_present_and_sane():
    # The fine-tuning target model and dataset are fixed by the thesis design.
    assert config.BASE_MODEL_ID == "google/paligemma-3b-pt-448"
    assert config.DATASET_ID == "Teklia/IAM-line"
    # 448px is the resolution the *-448 checkpoint expects; mismatch breaks the vision tower.
    assert config.IMAGE_SIZE == 448
    # A non-empty transcription prompt prefix is required for PaliGemma conditioning.
    assert isinstance(config.TRANSCRIPTION_PROMPT, str) and config.TRANSCRIPTION_PROMPT


def test_set_seed_is_deterministic():
    # Same seed -> same random draw. This guards reproducibility claims in the thesis.
    config.set_seed(123)
    first = [random.random() for _ in range(3)]
    config.set_seed(123)
    second = [random.random() for _ in range(3)]
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.config'`

- [ ] **Step 3: Write minimal implementation**

`src/htr_sp1/config.py`:
```python
"""Central configuration for SP-1.

Every tunable number, path, and identifier lives here so the rest of the code never
hard-codes values (DRY). When you change a hyperparameter for an experiment, you change
it in ONE place and the whole pipeline + notebook follow.
"""
from __future__ import annotations

import os
import random

# --- Fixed design choices (do not change without revising the thesis methodology) ---

# The exact base checkpoint we fine-tune. The "-448" suffix means it expects 448px images.
BASE_MODEL_ID = "google/paligemma-3b-pt-448"

# Official IAM line-level dataset on the Hub, with train/validation/test splits.
DATASET_ID = "Teklia/IAM-line"

# The vision tower of the -448 checkpoint is trained for 448x448 inputs. The processor
# resizes for us, but we keep the number here so it is explicit and testable.
IMAGE_SIZE = 448

# PaliGemma is *conditioned* on a text prompt. For pure transcription we use a short,
# fixed instruction. NOTE (flagged in the spec): confirm/tune this prefix against the
# PaliGemma task convention early — it is intentionally a single constant so tuning it
# is a one-line change.
TRANSCRIPTION_PROMPT = "transcribe the handwritten text\n"

# --- Training hyperparameters (safe starting points for a Colab T4 16GB) ---

# Per-device batch of 1 keeps memory low; we recover effective batch size via accumulation.
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8           # effective batch = 1 * 8 = 8
LEARNING_RATE = 2e-4                  # typical for LoRA adapters
NUM_TRAIN_EPOCHS = 3
MAX_TARGET_TOKENS = 64               # IAM lines are short; caps memory + generation length

# LoRA shape. Small rank keeps the adapter tiny and training cheap.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Attention + projection layers PaliGemma's language model exposes for LoRA.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Paths (Colab: point OUTPUT_DIR at a mounted Drive folder to survive disconnects) ---

# Default to a local folder; the notebook overrides this with a Google Drive path.
OUTPUT_DIR = os.environ.get("HTR_OUTPUT_DIR", "outputs/sp1")

# Where the exported model goes on the Hub. Override via env for your own account.
HF_HUB_REPO_ID = os.environ.get("HTR_HUB_REPO_ID", "your-username/paligemma-iam-line-qlora")

# --- Reproducibility ---

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we rely on so a run is reproducible.

    We seed Python's `random`, and (lazily, only if installed) numpy and torch. We import
    numpy/torch *inside* the function so unit tests on a laptop without them still pass.

    Args:
        seed: The integer seed. Defaults to the module-level SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # makes hash-based ordering deterministic too
    try:  # numpy/torch are heavy and may be absent in a minimal test env — degrade gracefully.
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/config.py tests/test_config.py
git commit -m "feat: add SP-1 config module with seed + pinned design constants

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Metrics module (CER / WER)

**Files:**
- Create: `src/htr_sp1/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
"""CER/WER are the headline numbers of the thesis, so we test exact, hand-computed cases."""
from htr_sp1 import metrics


def test_cer_perfect_match_is_zero():
    assert metrics.cer("the quick brown fox", "the quick brown fox") == 0.0


def test_cer_single_substitution_percentage():
    # "fox" -> "fux": 1 wrong char out of 19 reference chars = 1/19 * 100 ≈ 5.263.
    value = metrics.cer("the quick brown fox", "the quick brown fux")
    assert round(value, 2) == 5.26


def test_wer_one_wrong_word_of_four():
    # 1 wrong word / 4 reference words = 25%.
    value = metrics.wer("the quick brown fox", "the quick brown fux")
    assert round(value, 2) == 25.0


def test_metrics_return_percentages_not_fractions():
    # We standardize on PERCENT (0–100) so UI/report numbers match the PRD example.
    assert metrics.cer("ab", "xy") == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.metrics'`

- [ ] **Step 3: Write minimal implementation**

`src/htr_sp1/metrics.py`:
```python
"""Character and Word Error Rate, expressed as percentages (0–100).

Both are edit-distance (Levenshtein) based: the minimum number of insertions, deletions,
and substitutions to turn the prediction into the reference, divided by the reference
length. We delegate the math to `jiwer` (well-tested) and only convert to percent here so
our numbers line up with the PRD's example output (e.g. CER 5.26, WER 25.0).
"""
from __future__ import annotations

import jiwer


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate as a percentage.

    Args:
        reference: The ground-truth transcription.
        hypothesis: The model's predicted transcription.

    Returns:
        Edit distance over characters / reference character count, times 100.
    """
    # jiwer.cer returns a fraction in [0, ~]; multiply by 100 for a human-readable percent.
    return jiwer.cer(reference, hypothesis) * 100.0


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate as a percentage.

    Args:
        reference: The ground-truth transcription.
        hypothesis: The model's predicted transcription.

    Returns:
        Edit distance over words / reference word count, times 100.
    """
    return jiwer.wer(reference, hypothesis) * 100.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/metrics.py tests/test_metrics.py
git commit -m "feat: add CER/WER metrics (percent, Levenshtein via jiwer)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Data module (prompt + example building)

**Files:**
- Create: `src/htr_sp1/data.py`
- Modify: `tests/conftest.py` (create with shared fakes)
- Test: `tests/test_data.py`

- [ ] **Step 1: Create shared fakes in `tests/conftest.py`**

```python
"""Shared test doubles so unit tests never download models or hit a GPU.

FakeProcessor/FakeModel mimic just the slices of the transformers API our code touches.
This lets us test our *own* logic (prompt building, input assembly, decoding, aggregation)
deterministically on a laptop.
"""
import pytest


class FakeBatch(dict):
    """Stands in for a transformers BatchEncoding: a dict that also supports `.to(device)`."""

    def to(self, _device):
        return self  # no real tensors to move; just return self so call sites work.


class FakeProcessor:
    """Mimics PaliGemmaProcessor for the calls our code makes."""

    def __init__(self):
        self.last_call = None  # records the most recent kwargs so tests can assert on them.

    def __call__(self, text=None, images=None, suffix=None, return_tensors=None):
        # Record what we were asked to encode; return a minimal fake batch.
        self.last_call = {"text": text, "images": images, "suffix": suffix}
        return FakeBatch(input_ids=[[1, 2, 3]])

    def decode(self, _token_ids, skip_special_tokens=True):
        # Tests inject the desired decoded string via `self.next_decoded`.
        return getattr(self, "next_decoded", "decoded text")


class FakeModel:
    """Mimics a transformers model: `.generate(...)` returns fixed token ids."""

    def __init__(self, generated_ids=None):
        self._generated_ids = generated_ids or [[1, 2, 3, 4]]

    def generate(self, **_kwargs):
        return self._generated_ids


@pytest.fixture
def fake_processor():
    return FakeProcessor()


@pytest.fixture
def fake_model():
    return FakeModel()
```

- [ ] **Step 2: Write the failing test**

`tests/test_data.py`:
```python
"""Tests for prompt construction and turning an IAM record into a training example."""
from htr_sp1 import config, data


def test_build_prompt_uses_config_constant():
    # The prompt must come from config so experiments tune it in one place.
    assert data.build_prompt() == config.TRANSCRIPTION_PROMPT


def test_build_training_example_passes_image_prompt_and_label(fake_processor):
    # A fake "PIL image" — our code should hand it straight to the processor.
    record = {"image": object(), "text": "the quick brown fox"}
    data.build_training_example(record, fake_processor)
    call = fake_processor.last_call
    assert call["images"] is record["image"]
    assert call["text"] == config.TRANSCRIPTION_PROMPT
    # The label is supplied as `suffix` so PaliGemma's processor builds the loss labels.
    assert call["suffix"] == "the quick brown fox"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.data'`

- [ ] **Step 4: Write minimal implementation**

`src/htr_sp1/data.py`:
```python
"""Dataset loading and example construction for IAM-line.

Two responsibilities:
1. `load_iam_splits` — fetch the official train/validation/test splits from the Hub.
2. `build_prompt` / `build_training_example` — turn a raw record into the inputs the
   PaliGemma processor needs (image + prompt prefix + label suffix).

The example-building functions take the processor as an argument (dependency injection) so
they are unit-testable with a fake and never trigger a model download.
"""
from __future__ import annotations

from typing import Any, Dict

from . import config


def load_iam_splits():
    """Load the official IAM-line splits from the Hub.

    Imported lazily so importing this module on a minimal laptop (no `datasets`) is cheap
    and our prompt/example unit tests stay fast.

    Returns:
        A DatasetDict with "train", "validation", and "test" splits. Each record has an
        "image" (PIL.Image) and a "text" (ground-truth transcription) field.
    """
    from datasets import load_dataset

    # Returns all splits; we keep them together so the caller picks what it needs.
    return load_dataset(config.DATASET_ID)


def build_prompt() -> str:
    """Return the fixed transcription prompt prefix (sourced from config)."""
    return config.TRANSCRIPTION_PROMPT


def build_training_example(record: Dict[str, Any], processor) -> Dict[str, Any]:
    """Encode one IAM record into model inputs WITH labels, for supervised fine-tuning.

    PaliGemma's processor builds the training labels for us when we pass the target text as
    `suffix`: it appends the suffix after the prompt and masks the prompt tokens in the loss.

    Args:
        record: An IAM record with "image" (PIL) and "text" (ground truth) keys.
        processor: A PaliGemmaProcessor (or compatible fake in tests).

    Returns:
        The processor's encoded batch (input_ids/attention_mask/pixel_values/labels...).
    """
    return processor(
        text=config.TRANSCRIPTION_PROMPT,  # the conditioning prompt prefix
        images=record["image"],            # the handwriting line image
        suffix=record["text"],             # the ground truth -> becomes the labels
        return_tensors="pt",
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/htr_sp1/data.py tests/conftest.py tests/test_data.py
git commit -m "feat: add IAM-line loading + example building with test fakes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Model module (4-bit base + LoRA)

**Files:**
- Create: `src/htr_sp1/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
"""We can't load a 3B model on a laptop, so we test the *configuration assembly* that
governs the QLoRA setup. The actual load is exercised in Colab via the notebook.
"""
from htr_sp1 import config, model


def test_quant_config_is_4bit_nf4():
    # QLoRA requires 4-bit NF4 quantization of the frozen base weights.
    qc = model.build_quant_config()
    assert qc["load_in_4bit"] is True
    assert qc["bnb_4bit_quant_type"] == "nf4"


def test_lora_config_uses_config_constants():
    lc = model.build_lora_config()
    assert lc["r"] == config.LORA_R
    assert lc["lora_alpha"] == config.LORA_ALPHA
    assert lc["target_modules"] == config.LORA_TARGET_MODULES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.model'`

- [ ] **Step 3: Write minimal implementation**

`src/htr_sp1/model.py`:
```python
"""Model construction for QLoRA fine-tuning.

We split the heavy "load the real model" step from the lightweight "describe how to load
it" step. `build_quant_config` and `build_lora_config` return plain dicts that are trivial
to unit-test; `load_trainable_model` consumes them and does the real (GPU/Colab) work.
"""
from __future__ import annotations

from typing import Any, Dict

from . import config


def build_quant_config() -> Dict[str, Any]:
    """Describe the 4-bit (QLoRA) quantization of the frozen base model.

    NF4 is the 4-bit format from the QLoRA paper. Compute dtype is float16 because the Colab
    T4 (Turing) supports fp16 but NOT bf16 — using bf16 here would error on a T4. Returned as
    a dict so tests can assert on it without importing bitsandbytes.
    """
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",  # T4 supports fp16, not bf16
        "bnb_4bit_use_double_quant": True,  # extra memory saving, important on 16GB
    }


def build_lora_config() -> Dict[str, Any]:
    """Describe the LoRA adapter shape (sourced from config)."""
    return {
        "r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "target_modules": config.LORA_TARGET_MODULES,
        "task_type": "CAUSAL_LM",  # PaliGemma's text decoder is causal-LM style
    }


def load_trainable_model():
    """Load PaliGemma in 4-bit and attach a fresh LoRA adapter (Colab/GPU only).

    Heavy imports are local so this module imports instantly in unit tests. Steps:
      1. Quantize the base model to 4-bit NF4 (frozen).
      2. Prepare it for k-bit training (enables gradient checkpointing, casts norms).
      3. Wrap it with a LoRA adapter — only the small adapter weights will train.

    Returns:
        (model, processor): the PEFT-wrapped model and its PaliGemmaProcessor.
    """
    import torch
    from transformers import (
        BitsAndBytesConfig,
        PaliGemmaForConditionalGeneration,
        PaliGemmaProcessor,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Translate our plain-dict config into the real bitsandbytes config object.
    qc = build_quant_config()
    bnb = BitsAndBytesConfig(
        load_in_4bit=qc["load_in_4bit"],
        bnb_4bit_quant_type=qc["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qc["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qc["bnb_4bit_use_double_quant"],
    )

    processor = PaliGemmaProcessor.from_pretrained(config.BASE_MODEL_ID)
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb,
        device_map="auto",  # let accelerate place layers on the single T4
    )
    # Required before adding LoRA on a quantized model: enables grad checkpointing etc.
    base = prepare_model_for_kbit_training(base)

    lc = build_lora_config()
    lora = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"],
        task_type=lc["task_type"],
    )
    model = get_peft_model(base, lora)  # only LoRA params are now trainable
    return model, processor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/model.py tests/test_model.py
git commit -m "feat: add QLoRA model assembly (4-bit NF4 base + LoRA adapter)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Inference interface (the SP-2 contract)

**Files:**
- Create: `src/htr_sp1/inference.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

`tests/test_inference.py`:
```python
"""`generate_transcription` is the public interface SP-2 will import. We test that it:
  - encodes the image + prompt,
  - calls the model's generate,
  - decodes and strips whitespace.
Using fakes keeps it laptop-fast and deterministic.
"""
from htr_sp1 import inference


def test_generate_transcription_returns_clean_text(fake_model, fake_processor):
    fake_processor.next_decoded = "  the quick brown fox  "  # decode() will return this
    out = inference.generate_transcription(fake_model, fake_processor, image=object())
    # Output is stripped so downstream CER/WER aren't polluted by padding whitespace.
    assert out == "the quick brown fox"


def test_generate_transcription_feeds_prompt_and_image(fake_model, fake_processor):
    img = object()
    inference.generate_transcription(fake_model, fake_processor, image=img)
    from htr_sp1 import config
    assert fake_processor.last_call["images"] is img
    assert fake_processor.last_call["text"] == config.TRANSCRIPTION_PROMPT
    # At inference there is NO suffix (we have no label; the model must produce the text).
    assert fake_processor.last_call["suffix"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.inference'`

- [ ] **Step 3: Write minimal implementation**

`src/htr_sp1/inference.py`:
```python
"""The public inference interface for the fine-tuned model.

THIS IS THE CONTRACT SP-2 CONSUMES. Keep it tiny and stable: give it a loaded model, a
processor, and a PIL image, and it returns the predicted transcription string. M1 (baseline)
in SP-2 is literally this call; M2 (CoT) will reuse the same model with a different prompt.
"""
from __future__ import annotations

from . import config


def generate_transcription(model, processor, image, max_new_tokens: int = config.MAX_TARGET_TOKENS) -> str:
    """Transcribe a single handwriting-line image.

    Args:
        model: A loaded PaliGemma model (fine-tuned, or base+adapter, or a test fake).
        processor: The matching PaliGemmaProcessor (or a test fake).
        image: A PIL.Image of one handwriting line.
        max_new_tokens: Generation cap; IAM lines are short so the default is small.

    Returns:
        The predicted transcription, whitespace-stripped.
    """
    # Encode WITHOUT a suffix — at inference we have no ground truth; the model generates it.
    inputs = processor(
        text=config.TRANSCRIPTION_PROMPT,
        images=image,
        suffix=None,
        return_tensors="pt",
    )
    # `.to(...)` is a no-op on fakes; on a real run it moves tensors to the model's device.
    inputs = inputs.to(getattr(model, "device", "cpu"))

    # Greedy decode (do_sample defaults False) for reproducible, deterministic transcriptions.
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode the first (only) sequence and strip padding/whitespace for clean metrics.
    text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/inference.py tests/test_inference.py
git commit -m "feat: add generate_transcription — the SP-2 inference contract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Training module (TrainingArguments + checkpoint/resume)

**Files:**
- Create: `src/htr_sp1/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:
```python
"""Test the TrainingArguments assembly — the knobs that make training fit a T4 and survive
Colab disconnects. The real fit() runs in the notebook.
"""
from htr_sp1 import config, train


def test_training_args_fit_t4_and_checkpoint():
    args = train.build_training_args(output_dir="/tmp/run")
    assert args["output_dir"] == "/tmp/run"
    assert args["per_device_train_batch_size"] == config.PER_DEVICE_TRAIN_BATCH_SIZE
    assert args["gradient_accumulation_steps"] == config.GRAD_ACCUMULATION_STEPS
    # Gradient checkpointing trades compute for memory — required on 16GB.
    assert args["gradient_checkpointing"] is True
    # We must save checkpoints so a Colab disconnect can resume (not lose hours of training).
    assert args["save_strategy"] == "epoch"
    # Evaluate each epoch so we can watch val-CER and stop when it stabilizes.
    assert args["evaluation_strategy"] == "epoch"


def test_find_resume_checkpoint_prefers_existing(tmp_path):
    # No checkpoints yet -> None (start fresh).
    assert train.find_resume_checkpoint(str(tmp_path)) is None
    # Create a checkpoint dir -> it should be returned so training resumes from it.
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    assert train.find_resume_checkpoint(str(tmp_path)) == str(ckpt)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.train'`

- [ ] **Step 3: Write minimal implementation**

`src/htr_sp1/train.py`:
```python
"""Training orchestration.

Split into testable pieces:
  - `build_training_args` returns a plain dict of HF TrainingArguments fields (T4-friendly,
    checkpoint-on-epoch) that we can assert on without importing transformers.
  - `find_resume_checkpoint` scans the output dir so a Colab disconnect resumes instead of
    restarting — central to the "survive disconnects" requirement.
  - `run_training` wires everything to a real Trainer (Colab/GPU only).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from . import config


def build_training_args(output_dir: str) -> Dict[str, Any]:
    """Return TrainingArguments fields tuned for a Colab T4 with frequent checkpoints.

    Args:
        output_dir: Where checkpoints/logs are written (point at Drive in Colab).
    """
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRAD_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "num_train_epochs": config.NUM_TRAIN_EPOCHS,
        "gradient_checkpointing": True,   # memory-for-compute trade; needed on 16GB
        "fp16": True,                     # T4 (Turing) supports fp16, not bf16
        "save_strategy": "epoch",         # checkpoint every epoch -> resumable
        "evaluation_strategy": "epoch",   # val metrics every epoch -> watch val-CER
        "logging_steps": 25,
        "report_to": "none",              # no external trackers; keep the thesis run simple
    }


def find_resume_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path of the latest checkpoint to resume from, or None to start fresh.

    HF Trainer writes folders named `checkpoint-<step>`. After a Colab disconnect we want to
    pick up where we left off rather than retrain from scratch.

    Args:
        output_dir: The training output directory (possibly on Drive).
    """
    if not os.path.isdir(output_dir):
        return None
    # Collect checkpoint dirs and choose the one with the highest step number.
    checkpoints = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not checkpoints:
        return None
    # Sort by the trailing integer step so "checkpoint-100" beats "checkpoint-25".
    return max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[-1]))


def run_training(model, processor, train_ds, eval_ds, output_dir: str):
    """Run the real fine-tuning loop on Colab/GPU, resuming if a checkpoint exists.

    Heavy imports are local so the module stays laptop-importable for unit tests.

    Args:
        model: PEFT-wrapped PaliGemma from `model.load_trainable_model`.
        processor: The matching processor.
        train_ds / eval_ds: HF datasets yielding {"image", "text"} records.
        output_dir: Checkpoint/log directory (Drive path in Colab).

    Returns:
        The trained model (LoRA weights updated in place).
    """
    from transformers import Trainer, TrainingArguments

    from .data import build_training_example

    def collate(batch):
        # Encode each record (image + prompt + label suffix) and let the processor build
        # the padded tensors + masked labels. One example per record keeps memory predictable.
        examples = [build_training_example(r, processor) for r in batch]
        return processor.tokenizer.pad(examples, return_tensors="pt") if hasattr(processor, "tokenizer") else examples[0]

    args = TrainingArguments(**build_training_args(output_dir))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )
    # Resume from the latest checkpoint if Colab disconnected mid-run; else start fresh.
    trainer.train(resume_from_checkpoint=find_resume_checkpoint(output_dir))
    return model
```

> **Note for the implementer:** the `collate` function above is the one place that depends on
> exact PaliGemma batch shapes. Validate it in the notebook on a 2-record batch *before* the
> full run (see notebook Step "sanity overfit"). If padding differs, adjust collate there —
> the unit-tested pieces (`build_training_args`, `find_resume_checkpoint`) are unaffected.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/train.py tests/test_train.py
git commit -m "feat: add training args + checkpoint-resume logic for Colab T4

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Evaluate module (CER/WER over a split)

**Files:**
- Create: `src/htr_sp1/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:
```python
"""Evaluation aggregates per-sample CER/WER into the headline test numbers. We inject a fake
transcriber so the test is deterministic and needs no model.
"""
from htr_sp1 import evaluate


def test_evaluate_split_averages_cer_and_wer():
    # Two records: first predicted perfectly, second has one wrong char ("fux").
    records = [
        {"image": object(), "text": "the quick brown fox"},
        {"image": object(), "text": "the quick brown fox"},
    ]
    # Fake transcriber: perfect on the first call, "fux" on the second.
    predictions = iter(["the quick brown fox", "the quick brown fux"])

    def fake_transcribe(image):
        return next(predictions)

    report = evaluate.evaluate_split(records, fake_transcribe)
    # Mean CER = (0 + 5.26) / 2 ≈ 2.63 ; mean WER = (0 + 25) / 2 = 12.5
    assert round(report["mean_cer"], 2) == 2.63
    assert round(report["mean_wer"], 2) == 12.5
    assert report["num_samples"] == 2
    assert len(report["per_sample"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.evaluate'`

- [ ] **Step 3: Write minimal implementation**

`src/htr_sp1/evaluate.py`:
```python
"""Evaluate the fine-tuned model on a dataset split and produce the thesis baseline numbers.

`evaluate_split` takes a `transcribe` callable (so tests inject a fake and the notebook
injects the real `generate_transcription` bound to the model/processor). It returns mean
CER/WER plus per-sample rows for the appendix table.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

from . import metrics


def evaluate_split(records: Iterable[Dict[str, Any]], transcribe: Callable[[Any], str]) -> Dict[str, Any]:
    """Compute mean and per-sample CER/WER over a split.

    Args:
        records: Iterable of {"image", "text"} records (e.g. the IAM test split).
        transcribe: Function mapping an image to a predicted string. In Colab this is
            `lambda img: generate_transcription(model, processor, img)`; in tests it's a fake.

    Returns:
        Dict with keys: mean_cer, mean_wer, num_samples, per_sample (list of row dicts).
    """
    per_sample = []
    for record in records:
        prediction = transcribe(record["image"])
        ground_truth = record["text"]
        # Per-sample errors feed both the average and the appendix table for Bab 4.
        per_sample.append(
            {
                "ground_truth": ground_truth,
                "prediction": prediction,
                "cer": metrics.cer(ground_truth, prediction),
                "wer": metrics.wer(ground_truth, prediction),
            }
        )

    num = len(per_sample)
    if num == 0:  # guard against an empty split so we never divide by zero.
        return {"mean_cer": 0.0, "mean_wer": 0.0, "num_samples": 0, "per_sample": []}

    return {
        "mean_cer": sum(row["cer"] for row in per_sample) / num,
        "mean_wer": sum(row["wer"] for row in per_sample) / num,
        "num_samples": num,
        "per_sample": per_sample,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/evaluate.py tests/test_evaluate.py
git commit -m "feat: add split evaluation aggregating mean + per-sample CER/WER

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Export module (merge + push to HF Hub)

**Files:**
- Create: `src/htr_sp1/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing test**

`tests/test_export.py`:
```python
"""Export pushes the adapter + merged weights to the Hub. We test the path/repo-id logic
and that push is invoked correctly, with the network call mocked.
"""
from unittest.mock import MagicMock

from htr_sp1 import export


def test_adapter_and_merged_repo_ids_are_distinct():
    base = "user/paligemma-iam-line-qlora"
    assert export.adapter_repo_id(base) == "user/paligemma-iam-line-qlora-adapter"
    assert export.merged_repo_id(base) == "user/paligemma-iam-line-qlora-merged"


def test_push_model_calls_push_to_hub(monkeypatch):
    model = MagicMock()
    processor = MagicMock()
    # Merging a PEFT model returns a plain model we then push.
    model.merge_and_unload.return_value = MagicMock()

    export.push_adapter(model, processor, repo_id="user/repo-adapter")
    model.push_to_hub.assert_called_once_with("user/repo-adapter", private=True)
    processor.push_to_hub.assert_called_once_with("user/repo-adapter", private=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.export'`

- [ ] **Step 3: Write minimal implementation**

`src/htr_sp1/export.py`:
```python
"""Export the fine-tuned model to the Hugging Face Hub.

We publish TWO artifacts (the deliverable for SP-2):
  - the LoRA *adapter* (tiny; pairs with the base model at load time), and
  - the *merged* fp16 weights (adapter folded into the base; self-contained).
Repo-id helpers keep naming consistent; push functions are thin wrappers over `push_to_hub`
so they're easy to test with mocks and easy to read.
"""
from __future__ import annotations


def adapter_repo_id(base_repo_id: str) -> str:
    """Hub repo id for the standalone LoRA adapter."""
    return f"{base_repo_id}-adapter"


def merged_repo_id(base_repo_id: str) -> str:
    """Hub repo id for the merged, self-contained fp16 model."""
    return f"{base_repo_id}-merged"


def push_adapter(model, processor, repo_id: str) -> None:
    """Push the LoRA adapter weights + processor to a private Hub repo.

    Args:
        model: The PEFT-wrapped model (only adapter weights are uploaded).
        processor: The processor, pushed alongside so loaders get matching preprocessing.
        repo_id: Target repo (use `adapter_repo_id(...)`).
    """
    model.push_to_hub(repo_id, private=True)      # private: thesis artifact, not public
    processor.push_to_hub(repo_id, private=True)


def push_merged(model, processor, repo_id: str) -> None:
    """Merge the LoRA adapter into the base and push the self-contained fp16 model.

    Merging makes a single model that needs no separate adapter at load time — convenient for
    SP-2's local runtime and for any future GGUF/MLX conversion.

    Args:
        model: The PEFT-wrapped model.
        processor: The matching processor.
        repo_id: Target repo (use `merged_repo_id(...)`).
    """
    merged = model.merge_and_unload()  # fold adapter into base weights, drop PEFT wrappers
    merged.push_to_hub(repo_id, private=True)
    processor.push_to_hub(repo_id, private=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the FULL suite to confirm nothing regressed**

Run: `python -m pytest`
Expected: PASS (all tests from Tasks 1–8 green)

- [ ] **Step 6: Commit**

```bash
git add src/htr_sp1/export.py tests/test_export.py
git commit -m "feat: add Hub export (adapter + merged) with mocked push tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Colab orchestration notebook (manual-run gate)

**Files:**
- Create: `notebooks/sp1_train.ipynb`

> This notebook is thin glue: it imports the tested package and runs the real GPU steps in
> order. It is NOT unit-tested; instead it has an explicit **sanity-overfit gate** before the
> full run. Build it cell-by-cell with a markdown header above each code cell explaining the
> "why" (the user reads this to understand and defend the pipeline).

- [ ] **Step 1: Create the notebook with the cells below**

Create `notebooks/sp1_train.ipynb` containing these cells in order (each code cell preceded by a markdown cell with the shown explanation):

1. **Markdown:** "## SP-1 Training — overview, hardware (T4), and what each section does."
2. **Code — install & mount:**
```python
# Install the exact pinned versions and mount Drive so checkpoints survive disconnects.
!pip install -q -r /content/requirements.txt
from google.colab import drive
drive.mount('/content/drive')
import os
# Point training output at Drive so a disconnect doesn't lose hours of work.
os.environ["HTR_OUTPUT_DIR"] = "/content/drive/MyDrive/htr_sp1/outputs"
os.environ["HTR_HUB_REPO_ID"] = "YOUR_HF_USERNAME/paligemma-iam-line-qlora"  # <-- set this
```
3. **Code — auth & path:**
```python
# Make the package importable, authenticate to the Hub, and seed everything.
import sys; sys.path.insert(0, "/content/src")
from huggingface_hub import login; login()  # paste an HF token with write access
from htr_sp1 import config, data, model as model_mod, train, inference, evaluate, export
config.set_seed()
```
4. **Code — load data:**
```python
# Load the official IAM-line splits; peek at one record to confirm image+text look right.
ds = data.load_iam_splits()
print(ds)
sample = ds["train"][0]; print(repr(sample["text"])); sample["image"]
```
5. **Code — load model:**
```python
# Load PaliGemma in 4-bit and attach the LoRA adapter. Print trainable params to confirm
# ONLY the small adapter trains (should be a tiny fraction of total params).
model, processor = model_mod.load_trainable_model()
model.print_trainable_parameters()
```
6. **Markdown:** "### Sanity gate — overfit 2 samples. If loss does NOT drop to ~0, STOP and fix `collate` before wasting T4 hours."
7. **Code — sanity overfit:**
```python
# Train on a tiny 2-record slice for a few steps; loss should collapse toward 0.
tiny = ds["train"].select(range(2))
_ = train.run_training(model, processor, tiny, tiny,
                       output_dir="/content/sanity_check")
# Manually confirm the printed training loss trended to ~0 before continuing.
```
8. **Code — full training:**
```python
# Full fine-tune on the official splits; auto-resumes from Drive if Colab disconnected.
model = train.run_training(model, processor, ds["train"], ds["validation"],
                           output_dir=os.environ["HTR_OUTPUT_DIR"])
```
9. **Code — evaluate test split:**
```python
# Bind the model into the inference contract and compute the baseline CER/WER (the M1 numbers).
transcribe = lambda img: inference.generate_transcription(model, processor, img)
report = evaluate.evaluate_split(ds["test"], transcribe)
print("TEST mean CER:", round(report["mean_cer"], 2), "  mean WER:", round(report["mean_wer"], 2))
```
10. **Code — save metrics for the thesis:**
```python
# Persist per-sample + summary metrics to Drive for the Bab 4 appendix table.
import json
with open("/content/drive/MyDrive/htr_sp1/test_metrics.json", "w") as f:
    json.dump(report, f, indent=2)
```
11. **Markdown:** "### Export — push adapter + merged to the Hub (private)."
12. **Code — export:**
```python
base = os.environ["HTR_HUB_REPO_ID"]
export.push_adapter(model, processor, export.adapter_repo_id(base))
export.push_merged(model, processor, export.merged_repo_id(base))
print("Pushed:", export.adapter_repo_id(base), "and", export.merged_repo_id(base))
```
13. **Markdown:** "### Validation gate — reload fresh from the Hub and transcribe a few test images. This is SP-1's definition of done."
14. **Code — reload & validate:**
```python
# Reload the MERGED model from the Hub (no local state) and confirm it transcribes sanely.
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
rid = export.merged_repo_id(base)
v_model = PaliGemmaForConditionalGeneration.from_pretrained(rid, device_map="auto")
v_proc = PaliGemmaProcessor.from_pretrained(rid)
for i in range(3):
    img = ds["test"][i]["image"]
    print("GT :", ds["test"][i]["text"])
    print("OUT:", inference.generate_transcription(v_model, v_proc, img))
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/sp1_train.ipynb
git commit -m "feat: add thin Colab orchestration notebook with sanity + validation gates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Documentation (README + inference contract for SP-2)

**Files:**
- Create: `README-sp1.md`

- [ ] **Step 1: Write `README-sp1.md`**

````markdown
# SP-1 — PaliGemma QLoRA for IAM-line Transcription

Fine-tunes `google/paligemma-3b-pt-448` with QLoRA on `Teklia/IAM-line` to transcribe
handwriting lines. Produced for the HTR thesis (see `docs/superpowers/specs/2026-06-08-sp1-model-training-design.md`).

## Layout
- `src/htr_sp1/` — tested, documented modules (config, data, metrics, model, inference, train, evaluate, export)
- `notebooks/sp1_train.ipynb` — Colab orchestration (run on a T4)
- `tests/` — laptop-runnable unit tests (no GPU/downloads)

## Run the tests locally (no GPU)
```bash
pip install pytest jiwer
python -m pytest
```

## Train in Colab
1. Upload `src/`, `requirements.txt` to the Colab session (or clone the repo).
2. Open `notebooks/sp1_train.ipynb`, set `HTR_HUB_REPO_ID` to your HF repo.
3. Run top to bottom. The **sanity gate** (overfit 2 samples) must show loss→~0 before the full run.
4. Outputs: test CER/WER (`test_metrics.json` on Drive) + adapter & merged repos on the Hub.

## Inference interface (the contract SP-2 imports)
```python
from htr_sp1.inference import generate_transcription
# model, processor: a loaded PaliGemma (merged repo) + its processor
text = generate_transcription(model, processor, pil_image)
```
- **Input:** loaded model, processor, one `PIL.Image` of a handwriting line.
- **Output:** predicted transcription `str` (whitespace-stripped).
- **M1** in SP-2 = this call. **M2 (CoT)** = same model, different prompt (SP-2 swaps the prompt).

## Definition of Done
- Full QLoRA run completed; val-CER stabilized.
- Test CER/WER reported.
- Adapter + merged pushed to the Hub (private).
- Fresh reload from the Hub transcribes sample images correctly (validation gate).
````

- [ ] **Step 2: Commit**

```bash
git add README-sp1.md
git commit -m "docs: add SP-1 README and SP-2 inference contract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Order matters:** Tasks 1–8 are pure TDD and run on a laptop. Task 9 (notebook) and the real training run on Colab. Do all unit tasks first; only then run the notebook.
- **The one risky integration point** is the `collate` function in `train.py` (exact PaliGemma batch shapes). The notebook's sanity-overfit gate exists precisely to catch this before a long run — do not skip it.
- **Prompt tuning:** `config.TRANSCRIPTION_PROMPT` is the single lever for the prompt convention the spec flagged. If transcriptions look wrong during the sanity gate, adjust it there.
- **Keep the comments.** They are a deliverable, not clutter.
```
