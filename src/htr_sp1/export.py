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


def push_merged(model, processor, repo_id: str) -> None:
    """Merge the LoRA adapter into the base and push the self-contained fp16 model.

    Merging makes a single model that needs no separate adapter at load time — convenient for
    SP-2's local runtime and for any future GGUF/MLX conversion.

    Args:
        model: The PEFT-wrapped model.
        processor: The matching processor.
        repo_id: Target repo (use `merged_repo_id(...)`).
    """
    merged = model.merge_and_unload()  # fold adapter into base weights, drop PEFT wrappers
    merged.push_to_hub(repo_id, private=True)
    processor.push_to_hub(repo_id, private=True)
