"""Model construction for QLoRA fine-tuning.

We split the heavy "load the real model" step from the lightweight "describe how to load
it" step. `build_quant_config` and `build_lora_config` return plain dicts that are trivial
to unit-test; `load_trainable_model` consumes them and does the real (GPU/Colab) work.
"""
from __future__ import annotations

from typing import Any, Dict

from . import config


def build_quant_config(compute_dtype: str = "float16") -> Dict[str, Any]:
    """Describe the 4-bit (QLoRA) quantization of the frozen base model.

    NF4 is the 4-bit format from the QLoRA paper. The default compute dtype is float16 because
    the Colab T4 (Turing) supports fp16 but NOT bf16 — using bf16 there would error. On an
    Ampere/Ada GPU (e.g. RunPod A5000) the caller passes "bfloat16" for faster, more stable
    compute. Returned as a dict so tests can assert on it without importing bitsandbytes.

    Args:
        compute_dtype: torch dtype name for 4-bit compute ("float16" or "bfloat16").
    """
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": compute_dtype,
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


def load_trainable_model(compute_dtype: str = "float16"):
    """Load PaliGemma in 4-bit and attach a fresh LoRA adapter (Colab/GPU only).

    Heavy imports are local so this module imports instantly in unit tests. Steps:
      1. Quantize the base model to 4-bit NF4 (frozen).
      2. Prepare it for k-bit training (enables gradient checkpointing, casts norms).
      3. Wrap it with a LoRA adapter — only the small adapter weights will train.

    Args:
        compute_dtype: 4-bit compute dtype ("float16" for T4, "bfloat16" for Ampere/Ada).

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
    qc = build_quant_config(compute_dtype)
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
    # With gradient checkpointing on a 4-bit PEFT model, the backward pass otherwise fails
    # with "element 0 of tensors does not require grad". This hook makes the (frozen) input
    # embeddings' outputs require grad so checkpointed activations have a grad path.
    base.enable_input_require_grads()

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
