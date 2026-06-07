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
