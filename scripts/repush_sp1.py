#!/usr/bin/env python
"""SP-1 recovery CLI: re-push a trained run's adapter + merged model to the Hub from disk.

Use this when training finished but the push failed (or to re-publish an existing run). It reads
the adapter from <output_dir>/final_adapter (falling back to the latest Trainer checkpoint),
builds the merged model only if <output_dir>/merged is missing, and pushes both to the Hub. No
retraining. HTR_PG_DSN/HTR_HUB_REPO_ID/HF_TOKEN are read from the shell or a local .env.

Usage:
    export HTR_HUB_REPO_ID="your-hf-username/paligemma-iam-line-qlora"
    huggingface-cli login                 # or set HF_TOKEN in .env
    python scripts/repush_sp1.py --output-dir outputs/sp1
"""
import argparse
import sys
from pathlib import Path

# src/ sits next to the repo root (one level up from scripts/). Prepend it so `htr_sp1`
# is importable without installing the package -- same approach as scripts/train_sp1.py.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# HTR_PG_DSN / HF_TOKEN are read from a local .env (if present) automatically -- htr_sp1.config
# calls load_dotenv() at import time, so no explicit dotenv loading is needed here.
from htr_sp1 import config, export, repush  # noqa: E402
from htr_sp1.train import find_resume_checkpoint  # noqa: E402


def main() -> None:
    import os

    p = argparse.ArgumentParser(description="Re-push a trained SP-1 run (adapter + merged) to the Hub.")
    p.add_argument(
        "--output-dir",
        default=os.environ.get("HTR_OUTPUT_DIR", config.OUTPUT_DIR),
        help="Run dir holding final_adapter/ and merged/ (default: $HTR_OUTPUT_DIR/config).",
    )
    p.add_argument(
        "--hub-repo",
        default=os.environ.get("HTR_HUB_REPO_ID", config.HF_HUB_REPO_ID),
        help="Base Hub repo id (default: $HTR_HUB_REPO_ID/config).",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="Explicit adapter dir/hub-id (default: <output_dir>/final_adapter or checkpoint).",
    )
    p.add_argument(
        "--compute-dtype",
        default="bfloat16",
        help="Full-precision dtype for the merge step (bfloat16 on Ampere/Ada).",
    )
    args = p.parse_args()

    # Delegate ALL branching to the pure, tested planning function. It resolves which adapter
    # to use (explicit > final_adapter > latest checkpoint), whether merged needs rebuilding,
    # and the target Hub repo ids -- without touching disk, models, or the network.
    plan = repush.resolve_repush_plan(
        args.output_dir, args.hub_repo,
        adapter=args.adapter, compute_dtype=args.compute_dtype,
        find_checkpoint=find_resume_checkpoint,
    )
    print(f"[SP-1 repush] adapter_source={plan.adapter_source}  need_merge={plan.need_merge}")

    if plan.need_merge:
        # Load the base at full precision on CPU, attach the adapter, merge, and save to disk.
        # This avoids the 4-bit rounding corruption (see export.merge_to_dir docstring).
        print(f"[SP-1 repush] building merged model -> {plan.merged_dir} ...")
        export.merge_to_dir(plan.adapter_source, plan.merged_dir, compute_dtype=plan.compute_dtype)
    else:
        # merged/ already exists on disk from the original training run; nothing to rebuild.
        print(f"[SP-1 repush] reusing existing merged dir {plan.merged_dir}")

    # Push adapter first (smaller; faster to detect auth/network issues early).
    # allow_patterns filters out optimizer/scheduler/rng state when the source is a raw checkpoint.
    print(f"[SP-1 repush] pushing adapter -> {plan.adapter_repo}")
    export.push_folder(plan.adapter_source, plan.adapter_repo,
                       allow_patterns=export.ADAPTER_ALLOW_PATTERNS)

    # Push merged model (larger; ~5.8 GB -- may take several minutes).
    print(f"[SP-1 repush] pushing merged  -> {plan.merged_repo}")
    export.push_folder(plan.merged_dir, plan.merged_repo)

    print("[SP-1 repush] done.")


if __name__ == "__main__":
    main()
