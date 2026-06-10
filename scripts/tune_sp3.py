#!/usr/bin/env python
"""SP-3 threshold tuning CLI: scan thresholds on validation and write the best one.

Needs (a) a populated pgvector store (run scripts/ingest_sp3.py first) and (b) M1 predictions on
the IAM validation split as a JSON list of {"prediction": ..., "ground_truth": ...}. Produce that
file from the SP-1 eval (scripts/eval_sp1.py writes per_sample rows you can adapt) or any M1 run.

Usage:
    # HTR_PG_DSN can come from the shell or a local .env file (see .env.example).
    export HTR_PG_DSN="postgresql://localhost:5432/htr"
    python scripts/tune_sp3.py --pairs val_m1_predictions.json --out tune_sp3.json
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# HTR_PG_DSN is read from a local .env (if present) automatically — htr_sp3.config calls
# load_dotenv() at import time, so no explicit loading is needed here.
from htr_sp1 import data  # noqa: E402
from htr_sp3 import tune, vocab  # noqa: E402
from htr_sp3.corrector import RagCorrector  # noqa: E402
from htr_sp3.store import PgVectorStore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Tune the SP-3 correction threshold on validation.")
    p.add_argument("--pairs", required=True, help="JSON list of {prediction, ground_truth}.")
    p.add_argument("--out", default="tune_sp3.json", help="Where to write the tuning result.")
    args = p.parse_args()

    pairs_raw = json.loads(Path(args.pairs).read_text())
    pairs = [(r["prediction"], r["ground_truth"]) for r in pairs_raw]

    # Vocab set (for the exact-match gate) from IAM train — same source as ingest.
    vocab_set = vocab.build_vocabulary(data.load_iam_splits()["train"])
    store = PgVectorStore()  # already populated by ingest_sp3.py

    def make_corrector(threshold: float) -> RagCorrector:
        return RagCorrector(store=store, vocab=vocab_set, threshold=threshold)

    grid = [round(0.10 + 0.05 * i, 2) for i in range(9)]  # 0.10 .. 0.50
    grid = [0.0] + grid                                   # include "no correction" baseline
    result = tune.tune_threshold(pairs, make_corrector, grid)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[SP-3 tune] best_threshold={result['best_threshold']} "
          f"best_cer={result['best_cer']:.2f} (baseline T=0.0 CER={result['curve'][0.0]:.2f})")
    print(f"[SP-3 tune] wrote {args.out}")


if __name__ == "__main__":
    main()
