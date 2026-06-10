"""Export the fine-tuned model to the Hugging Face Hub.

We publish TWO artifacts (the deliverable for SP-2):
  - the LoRA *adapter* (tiny; pairs with the base model at load time), and
  - the *merged* fp16 weights (adapter folded into the base; self-contained).
Repo-id helpers keep naming consistent. The full export pipeline is:
  save_adapter -> merge_to_dir -> push_folder (via save_and_push).
Artifacts are always written to disk BEFORE any network call so a push failure never loses
the run — the on-disk artifacts can be re-pushed later (scripts/repush_sp1.py).
"""
from __future__ import annotations


def adapter_repo_id(base_repo_id: str) -> str:
    """Hub repo id for the standalone LoRA adapter."""
    return f"{base_repo_id}-adapter"


def merged_repo_id(base_repo_id: str) -> str:
    """Hub repo id for the merged, self-contained fp16 model."""
    return f"{base_repo_id}-merged"


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


def merge_to_dir(adapter_source, merged_dir, *, compute_dtype="bfloat16"):
    """Merge a trained adapter into a full-precision base and write the self-contained model.

    CRITICAL (unchanged from the original push_merged rationale): we do NOT merge the in-memory
    4-bit model — folding the adapter into 4-bit weights rounds and visibly corrupts generations.
    Instead we load the base at full precision on CPU (no quantization, no device_map, so we do
    not fight any resident training model for GPU memory), attach the adapter from *adapter_source*,
    merge there, and save to disk. The processor is loaded from the base id (it is never
    fine-tuned) so this works whether *adapter_source* is a final_adapter dir or a raw checkpoint.

    Args:
        adapter_source: Path (or Hub id) of the trained LoRA adapter.
        merged_dir: Destination directory for the merged fp16 model + processor.
        compute_dtype: Full-precision dtype to load the base in ("bfloat16" on Ampere/Ada).

    Returns:
        merged_dir.
    """
    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from peft import PeftModel

    from . import config

    base = PaliGemmaForConditionalGeneration.from_pretrained(
        config.BASE_MODEL_ID, torch_dtype=getattr(torch, compute_dtype),
    )
    merged = PeftModel.from_pretrained(base, adapter_source).merge_and_unload()
    merged.save_pretrained(merged_dir)

    # Processor comes from the base model id (identical, never trained) so a checkpoint source
    # without a saved processor still produces a self-contained merged dir.
    processor = PaliGemmaProcessor.from_pretrained(config.BASE_MODEL_ID)
    processor.save_pretrained(merged_dir)
    return merged_dir


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


def save_and_push(model, processor, *, output_dir, hub_repo, compute_dtype="bfloat16", push=True):
    """Persist artifacts to disk, then (optionally) push them — the SP-1 definition of done.

    Order matters: the adapter and merged model are written to disk BEFORE any network call, so a
    push failure never loses the run — it can be re-pushed later from disk (scripts/repush_sp1.py).
    Only the push step may fail; it is caught and reported via the returned status.

    Args:
        model: The PEFT-wrapped model (adapter source).
        processor: The matching processor.
        output_dir: Run directory; artifacts go to <output_dir>/final_adapter and <output_dir>/merged.
        hub_repo: Base Hub repo id; adapter/merged are pushed to its -adapter / -merged repos.
        compute_dtype: Full-precision dtype for the merge ("bfloat16" on Ampere/Ada, "float16" on T4).
        push: When False (e.g. --no-push), write both dirs but skip the network entirely.

    Returns:
        {"adapter_dir": str, "merged_dir": str, "pushed": bool, "error": str | None}.
    """
    import os

    adapter_dir = save_adapter(model, processor, os.path.join(output_dir, "final_adapter"))
    merged_dir = merge_to_dir(adapter_dir, os.path.join(output_dir, "merged"),
                              compute_dtype=compute_dtype)

    status = {"adapter_dir": adapter_dir, "merged_dir": merged_dir, "pushed": False, "error": None}
    if not push:
        return status

    try:
        push_folder(adapter_dir, adapter_repo_id(hub_repo), allow_patterns=ADAPTER_ALLOW_PATTERNS)
        push_folder(merged_dir, merged_repo_id(hub_repo))
        status["pushed"] = True
    except Exception as exc:  # network/auth failure must not lose the on-disk run
        status["error"] = str(exc)
    return status
