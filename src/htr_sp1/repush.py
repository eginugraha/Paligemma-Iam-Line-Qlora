"""Plan a re-push (recovery) without touching the model, disk writes, or network.

`resolve_repush_plan` is pure logic: it decides WHICH adapter to push (an explicit override, else
the run's final_adapter, else the latest Trainer checkpoint), whether the merged model still needs
to be built, and the target repo ids. The thin CLI (scripts/repush_sp1.py) executes the plan with
the real export functions. Existence and checkpoint lookups are injected so this is unit-testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class RepushPlan:
    """The fully-resolved inputs for a re-push."""

    adapter_source: str   # dir (or hub id) of the adapter to push / merge from
    merged_dir: str       # where the merged model lives / will be written
    need_merge: bool      # True if merged_dir must be built before pushing
    adapter_repo: str     # target Hub repo for the adapter
    merged_repo: str      # target Hub repo for the merged model
    compute_dtype: str    # full-precision dtype for the merge


def resolve_repush_plan(
    output_dir: str,
    hub_repo: str,
    *,
    adapter: Optional[str] = None,
    compute_dtype: str = "bfloat16",
    exists: Callable[[str], bool] = os.path.exists,
    find_checkpoint: Callable[[str], Optional[str]] = None,
) -> RepushPlan:
    """Resolve where to read the adapter from and what still needs building/pushing.

    Adapter source precedence: explicit *adapter* > <output_dir>/final_adapter > latest Trainer
    checkpoint (via *find_checkpoint*). Raises FileNotFoundError if none exist.

    Args:
        output_dir: The run directory holding final_adapter/, merged/, checkpoint-*/.
        hub_repo: Base Hub repo id (adapter/merged repos are derived from it).
        adapter: Optional explicit adapter dir/hub-id override.
        compute_dtype: Full-precision dtype for the merge step.
        exists: Predicate for path existence (injected for tests).
        find_checkpoint: Returns the latest Trainer checkpoint dir or None (injected for tests).
    """
    from .export import adapter_repo_id, merged_repo_id

    final_adapter = os.path.join(output_dir, "final_adapter")
    merged_dir = os.path.join(output_dir, "merged")

    if adapter is not None:
        if not exists(adapter):
            raise FileNotFoundError(f"--adapter path does not exist: {adapter}")
        adapter_source = adapter
    elif exists(final_adapter):
        adapter_source = final_adapter
    else:
        checkpoint = find_checkpoint(output_dir) if find_checkpoint is not None else None
        if checkpoint is None:
            raise FileNotFoundError(
                f"No adapter found: neither {final_adapter} nor any Trainer checkpoint in "
                f"{output_dir}. Pass --adapter explicitly or re-run training."
            )
        adapter_source = checkpoint

    return RepushPlan(
        adapter_source=adapter_source,
        merged_dir=merged_dir,
        need_merge=not exists(merged_dir),
        adapter_repo=adapter_repo_id(hub_repo),
        merged_repo=merged_repo_id(hub_repo),
        compute_dtype=compute_dtype,
    )
