#!/usr/bin/env python
"""SP-1 standalone evaluation: re-measure CER/WER for base + adapter at a chosen precision.

WHY THIS EXISTS
The original baseline (CER ~17%) was measured with the base loaded in 4-bit (QLoRA's training
config). If we deploy a bf16/merged model instead, the base precision differs from what the
adapter was trained against, so the score can shift slightly. This script lets us re-evaluate
on the EXACT configuration we intend to ship, so the number we report matches the model we use.

It reuses the package's real evaluation code (`evaluate.evaluate_split` + `inference`), so the
methodology is identical to the training run -- only the base precision changes.

WHERE TO RUN
On a CUDA GPU machine (e.g. the A6000 used for training). It will NOT run on a Mac/CPU for the
4bit mode (bitsandbytes is CUDA-only), and full-precision modes on CPU would be far too slow.

USAGE
    export HTR_HUB_REPO_ID="your-hf-username/paligemma-iam-line-qlora"
    huggingface-cli login                      # base is gated; adapter repo is private
    # quick smoke test on 50 samples in bf16:
    python scripts/eval_sp1.py --base-precision bf16 --limit 50
    # full apple-to-apple comparison:
    python scripts/eval_sp1.py --base-precision 4bit --out test_metrics_4bit.json
    python scripts/eval_sp1.py --base-precision bf16 --out test_metrics_bf16.json
"""
import argparse
import json
import sys
from pathlib import Path

# src/ sits next to the repo root (one level up from scripts/). Prepend it so `htr_sp1`
# is importable without installing the package -- same approach as scripts/train_sp1.py.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import os  # noqa: E402

from htr_sp1 import config, data, evaluate, export, inference, model as model_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SP-1 base+adapter at a chosen precision.")
    p.add_argument(
        "--adapter", default=None,
        help="Adapter Hub repo id or local dir. Default: <HTR_HUB_REPO_ID>-adapter.",
    )
    p.add_argument(
        "--base-precision", choices=["4bit", "bf16", "fp32"], default="bf16",
        help="How to load the base: 4bit reproduces training; bf16 (default) is the deploy config.",
    )
    p.add_argument("--split", default="test", help="Dataset split to evaluate (default: test).")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N samples (quick smoke test). Default: the whole split.",
    )
    p.add_argument(
        "--out", default=None,
        help="Where to write metrics JSON. Default: test_metrics_<base-precision>.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve the adapter location: explicit flag, else derive from the project's hub repo env.
    adapter = args.adapter or export.adapter_repo_id(
        os.environ.get("HTR_HUB_REPO_ID", config.HF_HUB_REPO_ID)
    )
    out_path = args.out or f"test_metrics_{args.base_precision}.json"

    print(f"[SP-1 eval] adapter={adapter}  base_precision={args.base_precision}  "
          f"split={args.split}  limit={args.limit}")

    # 1. Load the data split (auto-downloads IAM from the Hub; cached after the first run).
    ds = data.load_iam_splits()[args.split]
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    # 2. Load base + adapter at the requested precision (no training, no merge).
    eval_model, processor = model_mod.load_eval_model(adapter, base_precision=args.base_precision)

    # 3. Bind an ALREADY-LOADED model into the transcribe closure (see evaluate_split's docstring:
    #    reloading inside the loop would be catastrophically slow).
    transcribe = lambda img: inference.generate_transcription(eval_model, processor, img)
    report = evaluate.evaluate_split(ds, transcribe)

    print(f"[SP-1 eval] {args.base_precision}: mean CER {report['mean_cer']:.2f}  "
          f"mean WER {report['mean_wer']:.2f}  over {report['num_samples']} samples")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SP-1 eval] wrote {out_path}")


if __name__ == "__main__":
    main()
