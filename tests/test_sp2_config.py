"""Config holds the knobs the backend reads at runtime. We assert the defaults and the
two engine names so a typo can't silently change behaviour."""
from htr_sp2 import config


def test_default_engine_is_fake():
    # Default to the GPU-free fake so tests and a fresh checkout work without RunPod creds.
    assert config.ENGINE == "fake"


def test_model_prompts_and_tags():
    # M1 reuses the SP-1 transcription prompt; M2 uses the CoT prompt.
    from htr_sp1 import config as sp1config
    from htr_sp2 import cot
    assert config.M1_PROMPT == sp1config.TRANSCRIPTION_PROMPT
    assert config.M2_PROMPT == cot.COT_PROMPT
    assert config.M1_STATUS_TAG == "Raw Output"
    assert config.M2_STATUS_TAG == "Reasoned"
    assert config.M2_MAX_NEW_TOKENS > config.M1_MAX_NEW_TOKENS
