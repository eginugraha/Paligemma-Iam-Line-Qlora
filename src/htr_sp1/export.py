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


def push_merged(model, processor, repo_id: str, *, compute_dtype: str = "bfloat16") -> None:
    """Merge the LoRA adapter into a FULL-PRECISION base and push the self-contained model.

    Merging makes a single model that needs no separate adapter at load time — convenient for
    SP-2's local runtime and for any future GGUF/MLX conversion.

    CRITICAL: we do NOT call `model.merge_and_unload()` on the in-memory model. During QLoRA the
    base is loaded in 4-bit (NF4), and folding the (higher-precision) adapter into 4-bit weights
    forces rounding that visibly corrupts generations (HF even warns about it). Instead we:
      1. dump ONLY the trained adapter to a scratch dir,
      2. reload the base at full precision (bf16) on CPU — CPU so we don't fight the still-resident
         4-bit training model for GPU memory during the push,
      3. attach the adapter to that clean base and merge there (no rounding loss).
    The result matches the base+adapter path used for evaluation, instead of the degraded 4-bit
    merge. See `model.load_eval_model` for the matching inference-time loader.

    Args:
        model: The PEFT-wrapped (4-bit base) model — used only as the SOURCE of the adapter.
        processor: The matching processor, pushed alongside.
        repo_id: Target repo (use `merged_repo_id(...)`).
        compute_dtype: Full-precision dtype to load the base in ("bfloat16" on Ampere/Ada).
    """
    import tempfile

    import torch
    from transformers import PaliGemmaForConditionalGeneration
    from peft import PeftModel

    from . import config

    with tempfile.TemporaryDirectory() as adapter_dir:
        # 1. Persist ONLY the trained adapter weights (tiny) to a scratch directory.
        model.save_pretrained(adapter_dir)

        # 2. Reload the base at full precision — NO quantization_config, so it is NOT 4-bit.
        #    No device_map -> loads on CPU, leaving the GPU free for the resident training model.
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            config.BASE_MODEL_ID, torch_dtype=getattr(torch, compute_dtype),
        )

        # 3. Attach the adapter to the clean base and fold it in — faithful, no rounding damage.
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    merged.push_to_hub(repo_id, private=True)
    processor.push_to_hub(repo_id, private=True)
