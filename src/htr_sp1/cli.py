"""Command-line orchestration for SP-1 training (the non-notebook entry point).

This module is the script equivalent of `notebooks/sp1_train.ipynb`: it runs the exact same
pipeline (set seed -> load data -> load 4-bit+LoRA model -> sanity gate -> full fine-tune ->
evaluate -> save metrics -> push to Hub -> reload-validation), but as a plain `.py` you can
run over SSH on a server (e.g. a RunPod A5000 Pod) without Jupyter or any Colab/Drive glue.

Design split (so most of it is unit-testable without a GPU):
  - `build_parser`, `resolve_precision`, `precision_to_settings`, `resolve_config` are PURE:
    they only compute configuration and are fully tested on a laptop.
  - `main` does the heavy GPU work (model load + training); it's thin glue over the already
    tested `htr_sp1` package, mirroring the notebook.

Config precedence for each value: CLI argument > environment variable > `config.py` default.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from . import config


# --- Pure configuration helpers (unit-tested) -----------------------------------------

def resolve_precision(choice: str) -> str:
    """Resolve a precision choice to a concrete mode.

    Args:
        choice: "auto", "bf16", or "fp16". "auto" probes the GPU via `config.detect_precision`.

    Returns:
        "bf16" or "fp16".
    """
    if choice == "auto":
        return config.detect_precision()
    return choice


def precision_to_settings(precision: str) -> dict:
    """Map a resolved precision to the two knobs the model/trainer need.

    Args:
        precision: "bf16" or "fp16".

    Returns:
        {"compute_dtype": <torch dtype name>, "bf16": <bool for TrainingArguments>}.
    """
    if precision == "bf16":
        return {"compute_dtype": "bfloat16", "bf16": True}
    return {"compute_dtype": "float16", "bf16": False}


@dataclass
class RunConfig:
    """The fully-resolved settings for one training run."""

    output_dir: str
    hub_repo: str
    epochs: int
    batch_size: int
    precision: str
    compute_dtype: str
    bf16: bool
    skip_sanity: bool
    no_push: bool
    no_eval: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser. All overrides default to None/False so the fallback
    chain (CLI > env > config) is applied later in `resolve_config`."""
    p = argparse.ArgumentParser(
        prog="train_sp1",
        description="Fine-tune PaliGemma with QLoRA on IAM-line (SP-1) — server/CLI entry point.",
    )
    p.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto",
                   help="Mixed precision. 'auto' picks bf16 on Ampere/Ada GPUs, else fp16.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of training epochs (default: config.NUM_TRAIN_EPOCHS).")
    p.add_argument("--output-dir", default=None,
                   help="Checkpoint/output dir (default: $HTR_OUTPUT_DIR or config.OUTPUT_DIR).")
    p.add_argument("--hub-repo", default=None,
                   help="Base HF Hub repo id (default: $HTR_HUB_REPO_ID or config.HF_HUB_REPO_ID).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-device train batch size (default: config.PER_DEVICE_TRAIN_BATCH_SIZE).")
    p.add_argument("--skip-sanity", action="store_true",
                   help="Skip the 2-sample overfit sanity gate before the full run.")
    p.add_argument("--no-push", action="store_true",
                   help="Do not push adapter/merged model to the Hub (and skip reload-validation).")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip test-split CER/WER evaluation and metrics export.")
    return p


def resolve_config(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None) -> RunConfig:
    """Apply the CLI > env > config precedence to produce a concrete `RunConfig`.

    Args:
        args: Parsed CLI namespace from `build_parser`.
        env: Environment mapping (defaults to `os.environ`); injectable for tests.
    """
    env = os.environ if env is None else env
    output_dir = args.output_dir or env.get("HTR_OUTPUT_DIR") or config.OUTPUT_DIR
    hub_repo = args.hub_repo or env.get("HTR_HUB_REPO_ID") or config.HF_HUB_REPO_ID
    epochs = args.epochs if args.epochs is not None else config.NUM_TRAIN_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else config.PER_DEVICE_TRAIN_BATCH_SIZE

    precision = resolve_precision(args.precision)
    settings = precision_to_settings(precision)

    return RunConfig(
        output_dir=output_dir,
        hub_repo=hub_repo,
        epochs=epochs,
        batch_size=batch_size,
        precision=precision,
        compute_dtype=settings["compute_dtype"],
        bf16=settings["bf16"],
        skip_sanity=args.skip_sanity,
        no_push=args.no_push,
        no_eval=args.no_eval,
    )


# --- Heavy pipeline (GPU; mirrors the notebook) ---------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> RunConfig:
    """Run the full SP-1 pipeline from the command line.

    Heavy imports stay inside this function so the module imports instantly for unit tests.
    Returns the resolved `RunConfig` (handy for logging/tests of the wiring).

    Args:
        argv: Optional argument list (defaults to `sys.argv[1:]`).
    """
    # Parse first so `--help` / bad args exit before any (heavier) submodule imports.
    args = build_parser().parse_args(argv)
    rc = resolve_config(args)

    import json

    from . import data, evaluate, export, inference
    from . import model as model_mod
    from . import train as train_mod

    # Apply runtime overrides so `build_training_args` (which reads config) picks them up.
    config.NUM_TRAIN_EPOCHS = rc.epochs
    config.PER_DEVICE_TRAIN_BATCH_SIZE = rc.batch_size
    config.set_seed()

    print(
        "[SP-1] run config: "
        f"precision={rc.precision} (compute_dtype={rc.compute_dtype}, bf16={rc.bf16}), "
        f"epochs={rc.epochs}, batch_size={rc.batch_size}, output_dir={rc.output_dir}, "
        f"hub_repo={rc.hub_repo}, skip_sanity={rc.skip_sanity}, no_push={rc.no_push}, "
        f"no_eval={rc.no_eval}"
    )

    # 1. Data
    ds = data.load_iam_splits()

    # 2. Model (4-bit + LoRA), precision-aware
    model, processor = model_mod.load_trainable_model(compute_dtype=rc.compute_dtype)

    # 3. Sanity gate — overfit 2 samples; loss must collapse toward 0 before the full run.
    if not rc.skip_sanity:
        print("[SP-1] sanity gate: overfitting 2 samples (loss should trend to ~0)...")
        tiny = ds["train"].select(range(2))
        train_mod.run_training(
            model, processor, tiny, tiny,
            output_dir=os.path.join(rc.output_dir, "sanity_check"), bf16=rc.bf16,
        )

    # 4. Full fine-tune (auto-resumes from a checkpoint in output_dir if present).
    print("[SP-1] full fine-tune starting...")
    model = train_mod.run_training(
        model, processor, ds["train"], ds["validation"],
        output_dir=rc.output_dir, bf16=rc.bf16,
    )

    # 5. Evaluate test split (the M1 baseline numbers) + persist metrics.
    if not rc.no_eval:
        transcribe = lambda img: inference.generate_transcription(model, processor, img)
        report = evaluate.evaluate_split(ds["test"], transcribe)
        print(f"[SP-1] TEST mean CER: {report['mean_cer']:.2f}  mean WER: {report['mean_wer']:.2f}")
        os.makedirs(rc.output_dir, exist_ok=True)
        with open(os.path.join(rc.output_dir, "test_metrics.json"), "w") as f:
            json.dump(report, f, indent=2)

    # 6. Export adapter + merged to the Hub, then reload-validate (definition of done).
    if not rc.no_push:
        export.push_adapter(model, processor, export.adapter_repo_id(rc.hub_repo))
        # Merge happens on a full-precision base (not the 4-bit one). Match the run's precision:
        # bf16 on Ampere/Ada, fp16 on a T4.
        export.push_merged(model, processor, export.merged_repo_id(rc.hub_repo),
                           compute_dtype="bfloat16" if rc.bf16 else "float16")
        print("[SP-1] pushed:", export.adapter_repo_id(rc.hub_repo),
              "and", export.merged_repo_id(rc.hub_repo))

        # Reload the MERGED model fresh from the Hub and transcribe a few test images.
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        rid = export.merged_repo_id(rc.hub_repo)
        v_model = PaliGemmaForConditionalGeneration.from_pretrained(rid, device_map="auto")
        v_proc = PaliGemmaProcessor.from_pretrained(rid)
        print("[SP-1] reload-validation:")
        for i in range(3):
            img = ds["test"][i]["image"]
            print("  GT :", ds["test"][i]["text"])
            print("  OUT:", inference.generate_transcription(v_model, v_proc, img))

    print("[SP-1] done.")
    return rc


if __name__ == "__main__":  # pragma: no cover - exercised via scripts/train_sp1.py
    main()
