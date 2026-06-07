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
