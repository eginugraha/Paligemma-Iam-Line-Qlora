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


def load_eval_model(adapter_source: str, *, base_precision: str = "bf16",
                    compute_dtype: str = "bfloat16"):
    """Load the base + a TRAINED LoRA adapter for inference/evaluation (no training).

    This is the honest way to re-measure CER/WER: we attach the already-trained adapter to a
    freshly loaded base and run generation. Nothing is trained or merged here, so the adapter
    (the training result) is untouched — we only vary HOW the base is loaded.

    The `base_precision` knob is the whole point of this function. The adapter was trained
    against a 4-bit base, so the precisions are NOT interchangeable and may shift the score:
      - "4bit": reload the base in 4-bit NF4, exactly like training time. This reproduces the
                original baseline numbers. Requires a CUDA GPU + bitsandbytes (won't run on Mac).
      - "bf16": load the base in bfloat16 (~5.8 GB). Full precision vs the coarse 4-bit; this
                is the config a merged-bf16 deployment effectively uses. Recommended default.
      - "fp32": load the base in float32 (~11.7 GB, the on-disk dtype). Most faithful, most
                memory; use only if you want zero precision compromise.

    Args:
        adapter_source: Hub repo id (e.g. "user/...-adapter") OR a local adapter directory.
        base_precision: One of "4bit" | "bf16" | "fp32" (see above).
        compute_dtype: 4-bit compute dtype, only used when base_precision == "4bit".

    Returns:
        (model, processor): the base+adapter model in eval mode and its PaliGemmaProcessor.
    """
    import torch
    from transformers import (
        BitsAndBytesConfig,
        PaliGemmaForConditionalGeneration,
        PaliGemmaProcessor,
    )
    from peft import PeftModel

    processor = PaliGemmaProcessor.from_pretrained(config.BASE_MODEL_ID)

    if base_precision == "4bit":
        # Same NF4 config as training so the base the adapter "sees" matches what it learned on.
        qc = build_quant_config(compute_dtype)
        bnb = BitsAndBytesConfig(
            load_in_4bit=qc["load_in_4bit"],
            bnb_4bit_quant_type=qc["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, qc["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=qc["bnb_4bit_use_double_quant"],
        )
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            config.BASE_MODEL_ID, quantization_config=bnb, device_map="auto",
        )
    elif base_precision in ("bf16", "fp32"):
        # No quantization: load the full-precision base, downcasting the on-disk float32 to the
        # requested dtype. bf16 is plenty precise next to 4-bit and halves memory vs fp32.
        dtype = torch.bfloat16 if base_precision == "bf16" else torch.float32
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            config.BASE_MODEL_ID, torch_dtype=dtype, device_map="auto",
        )
    else:
        raise ValueError(f"base_precision must be '4bit', 'bf16' or 'fp32', got {base_precision!r}")

    # Attach the trained adapter on top of the chosen base. No merge — adapter stays separate,
    # exactly the configuration used to produce the original evaluation numbers.
    model = PeftModel.from_pretrained(base, adapter_source)
    model.eval()  # disable dropout etc.; we are only generating, never training
    return model, processor
