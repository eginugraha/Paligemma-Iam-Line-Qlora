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


# Files that make up a self-contained, loadable adapter (LoRA weights + its config + the
# processor/tokenizer). When the push source is a FULL Trainer checkpoint (recovery fallback),
# this allowlist uploads ONLY these to the adapter repo and skips optimizer/scheduler/rng state.
ADAPTER_ALLOW_PATTERNS = [
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def save_adapter(model, processor, adapter_dir):
    """Write the trained LoRA adapter (+ processor) to *adapter_dir* and return the path.

    PEFT's save_pretrained writes ONLY the small adapter weights, not the frozen base. The
    processor is saved alongside so the directory is self-contained for loading/pushing.

    Args:
        model: The PEFT-wrapped model whose adapter weights are written.
        processor: The matching processor.
        adapter_dir: Destination directory (e.g. <output_dir>/final_adapter).
    """
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    return adapter_dir


def push_folder(local_dir, repo_id, *, commit_message=None, private=True, allow_patterns=None):
    """Upload a local directory to a Hub repo (the ONE place a push happens).

    Uploading to an existing repo adds a new commit (same-named files are overwritten, but the
    Hub keeps history, so it is reversible). No model is loaded into memory — we ship the bytes
    already written to disk, which is fast and cheap even for the ~5.8 GB merged model.

    Args:
        local_dir: Directory whose contents are uploaded (e.g. an adapter or merged-model dir).
        repo_id: Target repo id (use adapter_repo_id(...) / merged_repo_id(...)).
        commit_message: Optional message recorded on the Hub commit (handy for run provenance).
        private: Create the repo private (thesis artifact, not public) if it does not exist.
        allow_patterns: Optional list of globs to restrict which files are uploaded (used to push
                        ONLY adapter/processor files when the source is a full Trainer checkpoint).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    # exist_ok=True: re-pushing to an existing repo is normal (recovery), not an error.
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir, repo_id=repo_id, repo_type="model",
        commit_message=commit_message, allow_patterns=allow_patterns,
    )
