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


def test_quant_config_compute_dtype_defaults_to_float16():
    # Default preserves the T4-safe baseline: 4-bit compute in float16.
    qc = model.build_quant_config()
    assert qc["bnb_4bit_compute_dtype"] == "float16"


def test_quant_config_compute_dtype_override():
    # On an Ampere/Ada GPU the caller can request bfloat16 4-bit compute.
    qc = model.build_quant_config(compute_dtype="bfloat16")
    assert qc["bnb_4bit_compute_dtype"] == "bfloat16"
    # Overriding precision must not disturb the other QLoRA invariants.
    assert qc["load_in_4bit"] is True
    assert qc["bnb_4bit_quant_type"] == "nf4"
