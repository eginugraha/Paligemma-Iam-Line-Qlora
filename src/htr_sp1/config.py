"""Central configuration for SP-1.

Every tunable number, path, and identifier lives here so the rest of the code never
hard-codes values (DRY). When you change a hyperparameter for an experiment, you change
it in ONE place and the whole pipeline + notebook follow.
"""
from __future__ import annotations

import os
import random

# --- Load a local .env file (optional) -------------------------------------------------
# Per-machine overrides and secrets (e.g. HTR_HUB_REPO_ID, HTR_OUTPUT_DIR, and the HF_TOKEN
# that huggingface_hub reads when push_to_hub uploads the model) can live in a gitignored
# .env at the repo root instead of being exported in every shell. We load it HERE, at the top
# of config, so the os.environ.get(...) calls below already see those values. python-dotenv is
# optional and load_dotenv does NOT override real shell exports (override defaults to False),
# so a missing package or missing file simply falls back to the existing environment.
try:
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # parents[2] == repo root
except ImportError:
    pass

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
GRAD_ACCUMULATION_STEPS = 8           # effective batch = PER_DEVICE_TRAIN_BATCH_SIZE * 8
# Evaluation batch size. CRITICAL: if left unset, HF TrainingArguments defaults this to 8.
# During eval the model emits logits of shape [batch, seq_len, vocab], and PaliGemma's vocab is
# ~257k, so cross_entropy upcasts to fp32 and a batch of 8 allocates ~8 GiB in ONE go -> OOM on
# a 24GB GPU (this is what blew up at the first end-of-epoch eval). We keep eval at 1, matching
# training; eval has no .backward() so a single sample fits comfortably.
PER_DEVICE_EVAL_BATCH_SIZE = 1
LEARNING_RATE = 2e-4                  # typical for LoRA adapters
NUM_TRAIN_EPOCHS = 3
MAX_TARGET_TOKENS = 64               # IAM lines are short; caps memory + generation length

# LoRA shape. Small rank keeps the adapter tiny and training cheap.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# We adapt the language model's attention projections only (q/k/v/o_proj). We deliberately do
# NOT LoRA-adapt the vision tower or the multimodal projector in this baseline: it keeps the
# adapter small and avoids depending on exact internal module names that vary across
# transformers versions (a wrong name hard-crashes model load). Adapting the projector is a
# documented future lever for the thesis, not part of the baseline.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Paths (Colab: point OUTPUT_DIR at a mounted Drive folder to survive disconnects) ---

# Default to a local folder; the notebook overrides this with a Google Drive path.
OUTPUT_DIR = os.environ.get("HTR_OUTPUT_DIR", "outputs/sp1")

# Where the exported model goes on the Hub. Override via env for your own account.
HF_HUB_REPO_ID = os.environ.get("HTR_HUB_REPO_ID", "your-username/paligemma-iam-line-qlora")

# --- Reproducibility ---

SEED = 42


def detect_precision() -> str:
    """Pick the best training precision for the current GPU: "bf16" or "fp16".

    bfloat16 is faster and numerically more stable than float16, but only Ampere/Ada-class
    GPUs (and newer) support it — a Turing T4 does NOT. We probe torch at runtime so the same
    code auto-adapts when moved between machines (e.g. Colab T4 -> a RunPod A6000) without a
    code edit. torch is imported lazily so this module stays importable on a CPU-only laptop.

    Returns:
        "bf16" if a bfloat16-capable CUDA GPU is present, otherwise the safe default "fp16".
    """
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
    except ImportError:
        pass
    return "fp16"


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we rely on so a run is reproducible.

    We seed Python's `random`, and (lazily, only if installed) numpy and torch. We import
    numpy/torch *inside* the function so unit tests on a laptop without them still pass.

    Args:
        seed: The integer seed. Defaults to the module-level SEED.
    """
    random.seed(seed)
    # NOTE: setting this here only affects *child* processes, not the current interpreter
    # (CPython reads PYTHONHASHSEED at startup). For fully deterministic hash ordering,
    # launch with `PYTHONHASHSEED=<seed> python ...` in the shell.
    os.environ["PYTHONHASHSEED"] = str(seed)
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
