#!/usr/bin/env python
"""Generate M1 (baseline) predictions on the IAM *validation* split for SP-3 threshold tuning.

WHY THIS EXISTS
---------------
The SP-3 RAG corrector (M3/M4) currently *worsens* accuracy because its correction threshold
(`DEFAULT_THRESHOLD = 0.34`) was never tuned (see docs/sp3-rag-correction-investigation-2026-06-13.md).
`scripts/tune_sp3.py` finds the optimal threshold, but it needs raw, *uncorrected* M1 predictions
paired with ground truth as its tuning material:

    [{"prediction": "...", "ground_truth": "..."}, ...]

This script produces exactly that file. It runs the ALREADY-TRAINED model — no training happens
here. M1 = the SP-1 PaliGemma base (4-bit NF4) + LoRA adapter, which is already deployed on the
RunPod Serverless endpoint. We just call that endpoint for ~100 images; RunPod spins a GPU worker
up on demand and tears it down afterward (~5 min, small cost). Nothing is deployed or retrained.

WHY THE VALIDATION SPLIT (NOT TEST)
-----------------------------------
The threshold is a hyperparameter. Tuning it on the *validation* split keeps the *test* split
untouched for the final Chapter-4 numbers (anti-leakage). Using test here would bias the reported
result, so this script defaults to --split validation on purpose.

PREREQUISITES
-------------
The RunPod engine reads its endpoint id / api key from the environment (or a local .env file —
htr_sp2.config calls load_dotenv() at import). Set these before running:

    export HTR_RUNPOD_ENDPOINT_ID="..."   # RunPod dashboard > Serverless > Endpoints
    export HTR_RUNPOD_API_KEY="..."
    # HTR_ENGINE is irrelevant here: we ask get_engine("runpod") explicitly.

USAGE
-----
    python scripts/gen_val_predictions.py                       # 100 validation samples, default out
    python scripts/gen_val_predictions.py --limit 200 --out val_m1_predictions.json
    python scripts/tune_sp3.py --pairs val_m1_predictions.json --out tune_sp3.json   # next step
"""
import argparse
import json
import sys
from pathlib import Path

# src/ sits one level up from scripts/. Prepend it so the htr_* packages import without
# installing the project -- same convention as scripts/eval_sp1.py and scripts/tune_sp3.py.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htr_sp1 import data  # noqa: E402
from htr_sp2 import config  # noqa: E402  (M1 prompt + token cap live here)
from htr_sp2.engine import EngineError, get_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate M1 baseline predictions on the IAM validation split (via RunPod)."
    )
    p.add_argument(
        "--split", default="validation",
        help="Dataset split to run on. Default: validation (keep 'test' untouched for final numbers).",
    )
    p.add_argument(
        "--limit", type=int, default=100,
        help="How many samples to predict. Default: 100 (enough to tune the threshold cheaply).",
    )
    p.add_argument(
        "--out", default="val_m1_predictions.json",
        help="Where to write the [{prediction, ground_truth}] list. Default: val_m1_predictions.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load the requested split (auto-downloads IAM from the Hub; cached after first run).
    #    Each record exposes a PIL image under "image" and the transcription under "text" --
    #    same field names htr_sp1.evaluate relies on.
    ds = data.load_iam_splits()[args.split]
    ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)

    # 2. Build the RunPod engine explicitly (don't depend on the HTR_ENGINE env default,
    #    which is "fake"). get_engine reads RUNPOD_* credentials from config/.env.
    engine = get_engine("runpod")
    print(f"[gen-val] split={args.split}  samples={n}  engine=runpod  -> {args.out}")

    # 3. Run M1 on each image. M1 = baseline prompt + SP-1 token cap; NO RAG correction here,
    #    because the tuner needs the *raw* predictions to simulate correction at each threshold.
    pairs: list[dict[str, str]] = []
    for i, record in enumerate(ds, start=1):
        try:
            prediction = engine.run(
                record["image"],
                config.M1_PROMPT,
                config.M1_MAX_NEW_TOKENS,
            )
        except EngineError as exc:
            # One bad call (timeout, cold-start hiccup) shouldn't throw away the whole run.
            # Record an empty prediction so the pair count still lines up, and keep going.
            print(f"[gen-val] sample {i}/{n} FAILED: {exc}", file=sys.stderr)
            prediction = ""

        pairs.append({"prediction": prediction, "ground_truth": record["text"]})

        # Lightweight progress so a 5-minute GPU run isn't a silent black box.
        if i % 10 == 0 or i == n:
            print(f"[gen-val] {i}/{n} done")

    # 4. Write the bare JSON list in exactly the shape tune_sp3.py expects.
    Path(args.out).write_text(json.dumps(pairs, indent=2))
    print(f"[gen-val] wrote {len(pairs)} pairs to {args.out}")
    print(f"[gen-val] next: python scripts/tune_sp3.py --pairs {args.out} --out tune_sp3.json")


if __name__ == "__main__":
    main()
