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
