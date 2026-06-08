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


def test_set_seed_numpy_deterministic():
    # Reproducibility that actually matters for training comes from numpy/torch, not just
    # Python's random. Skip cleanly on a minimal env where numpy isn't installed.
    import pytest

    np = pytest.importorskip("numpy")
    config.set_seed(123)
    first = np.random.rand(3).tolist()
    config.set_seed(123)
    second = np.random.rand(3).tolist()
    assert first == second


def test_detect_precision_returns_known_value():
    # Auto-precision must always resolve to one of the two supported modes so callers can
    # map it deterministically. On the CPU-only test machine (no CUDA) it must be "fp16",
    # never "bf16" — picking bf16 without a capable GPU would crash a real run.
    assert config.detect_precision() in ("bf16", "fp16")


def test_detect_precision_is_fp16_without_cuda():
    # No torch or no CUDA -> the safe default is fp16 (works on the widest set of GPUs,
    # including the Colab T4 the baseline was tuned for).
    import pytest

    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this test pins the no-CUDA fallback only.")
    assert config.detect_precision() == "fp16"
