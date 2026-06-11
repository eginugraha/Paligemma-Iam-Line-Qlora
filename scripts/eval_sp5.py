#!/usr/bin/env python
"""SP-5 batch evaluation CLI: run M1-M4 over IAM-test and store results for the dashboard.

WHERE TO RUN
On a CUDA GPU machine (needs the real engine). Set ENGINE=runpod (+ its env) or run where the
local engine is available. Requires HTR_PG_DSN; use --rag to also evaluate M3/M4 (needs
HTR_ENABLE_RAG infra / pgvector ingested).

IAM dataset field names (confirmed from src/htr_sp1/data.py)
-------------------------------------------------------------
The ``load_dataset("iamdb/iam_handwriting", ...)`` records have exactly two content fields:
  - ``"image"``  — a PIL.Image (typically grayscale mode "L"; ``ensure_rgb`` converts to RGB).
  - ``"text"``   — the ground-truth transcription string for the line.
There is NO ``"id"`` field in the raw HuggingFace dataset.  We therefore synthesise a
``sample_id`` from the split-local integer index (``str(i)``), which is stable as long as the
dataset version does not change and the iteration order is deterministic — both guaranteed by the
official IAM split on the Hub.  This means the ``sample_id`` column in the dashboard will show
``"0"``, ``"1"``, … rather than a human-readable IAM filename.  For the thesis comparison tables
this is sufficient; if a human-readable slug is needed later, iterate the dataset with
``.features`` or add an ``"id"`` field via a HuggingFace dataset script.

USAGE
    python scripts/eval_sp5.py --limit 200
    python scripts/eval_sp5.py --limit 200 --rag --model-ref eginugraha/paligemma-iam-line-qlora
"""
import argparse
import sys
from pathlib import Path

# src/ sits one level above scripts/ in the repository layout.  Prepend it so ``htr_sp1``,
# ``htr_sp2``, and ``htr_sp5`` are importable without installing the packages — matching
# the convention used in scripts/eval_sp1.py and scripts/train_sp1.py.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htr_sp1 import data as iam_data            # noqa: E402
from htr_sp2.engine import get_engine           # noqa: E402
from htr_sp2.corrector_factory import get_corrector  # noqa: E402
from htr_sp5.evalrun import run_eval            # noqa: E402
from htr_sp5.store import Sp5Store              # noqa: E402


def _load_samples(limit: int) -> list[dict]:
    """Load samples from the IAM test split, capped at ``limit``, and return them as dicts.

    Each returned dict has the three keys that ``run_eval`` expects:
      - ``"sample_id"``    — synthesised from the split-local index (see module docstring).
      - ``"image"``        — PIL.Image converted to RGB via ``ensure_rgb``.
      - ``"ground_truth"`` — the ``"text"`` field from the IAM record.

    The conversion step (grayscale "L" → RGB) is mandatory: PaliGemma's image processor
    raises "Unsupported number of image dimensions: 2" on a 2D array, so all IAM images
    must be 3-channel before reaching the engine.

    Args:
        limit: Maximum number of samples to return.  Pass a small value (e.g. 5) for a quick
               smoke test; pass a large value (e.g. 2000) for a full evaluation run.

    Returns:
        A list of sample dicts, length ``min(limit, len(test_split))``.
    """
    # load_iam_splits() returns a DatasetDict with "train", "validation", and "test" keys.
    # Each record in the "test" split has "image" (PIL) and "text" (str) fields.
    # NOTE: Field names confirmed from src/htr_sp1/data.py build_training_example() which
    # uses record["image"] and record["text"] — those are the canonical IAM column names.
    splits = iam_data.load_iam_splits()
    test = splits["test"]

    out = []
    for i, rec in enumerate(test):
        if i >= limit:
            break
        out.append({
            # No "id" field in the IAM dataset; use the split-local index as a stable
            # string key.  See module docstring for full rationale.
            "sample_id": str(i),
            # ensure_rgb converts IAM's grayscale "L" images to 3-channel RGB as required
            # by PaliGemma.  Calling it here means the engine never sees a 2D image array.
            "image": iam_data.ensure_rgb(rec["image"]),
            # "text" is the ground-truth transcription column in the IAM HuggingFace dataset.
            # Renamed to "ground_truth" here to match run_eval's sample dict contract.
            "ground_truth": rec["text"],
        })
    return out


def main() -> None:
    """Parse CLI arguments, wire real collaborators, and run the batch evaluation."""
    p = argparse.ArgumentParser(description="SP-5 batch evaluation over IAM-test.")
    p.add_argument(
        "--limit", type=int, default=200,
        help="Number of IAM-test samples to evaluate (default: 200).",
    )
    p.add_argument(
        "--rag", action="store_true",
        help=(
            "Also evaluate M3/M4 (requires HTR_ENABLE_RAG=true in the environment "
            "and pgvector ingested; without it the corrector will be None and the "
            "run falls back to M1/M2 only)."
        ),
    )
    p.add_argument(
        "--model-ref", default=None,
        help="Model/adapter version label stored on the run (e.g. a HuggingFace repo id).",
    )
    p.add_argument(
        "--dataset", default="iam-line-test",
        help="Human-readable dataset label stored on the eval_run row (default: iam-line-test).",
    )
    args = p.parse_args()

    # --- 1. Load the IAM test samples (downloads from Hub on first run, cached after) ----
    print(f"[SP-5 eval] loading {args.limit} samples from IAM test split …")
    samples = _load_samples(args.limit)
    print(f"[SP-5 eval] loaded {len(samples)} samples")

    # --- 2. Instantiate collaborators (engine, optional corrector, persistent store) ------
    # get_engine() reads the ENGINE env var ("runpod", "local", etc.) and returns the
    # appropriate InferenceEngine implementation.  Must be called on a CUDA machine.
    engine = get_engine()

    # get_corrector() is only called when --rag is set; it reads HTR_ENABLE_RAG and related
    # env vars to build the corrector.  If HTR_ENABLE_RAG is not set (or the pgvector store
    # is not reachable) get_corrector() returns None, which silently skips M3/M4.
    # We surface that case as an explicit warning so the user knows their --rag flag was
    # ignored rather than discovering it only from the absence of M3/M4 rows in the store.
    corrector = get_corrector() if args.rag else None
    if args.rag and corrector is None:
        # This is the "silent failure" guard: --rag was requested but no corrector was built.
        # Most common cause: HTR_ENABLE_RAG=true was not set in the environment, or the
        # pgvector store has not been ingested yet.  We warn on stderr (not stdout) so that
        # log-scraping pipelines that only capture stdout are not confused.
        print(
            "[SP-5 eval] WARNING: --rag was requested but no corrector was created. "
            "Did you set HTR_ENABLE_RAG=true in the environment? "
            "Is pgvector ingested? "
            "Proceeding with M1/M2 only.",
            file=sys.stderr,
        )
    elif corrector is not None:
        print("[SP-5 eval] RAG corrector loaded — M3/M4 will be evaluated")
    else:
        print("[SP-5 eval] RAG disabled — evaluating M1/M2 only")

    # Sp5Store handles create_eval_run + insert_eval_results.  create_schema() is idempotent
    # (uses CREATE TABLE IF NOT EXISTS) so it is safe to call on every run.
    store = Sp5Store()
    store.create_schema()

    # --- 3. Run the evaluation pipeline -----------------------------------------------
    print(f"[SP-5 eval] starting evaluation: dataset={args.dataset} model_ref={args.model_ref}")
    run_id = run_eval(
        samples, engine, corrector, store,
        dataset=args.dataset,
        model_ref=args.model_ref,
    )

    # --- 4. Report completion ---------------------------------------------------------
    print(
        f"[SP-5 eval] done — eval_run {run_id} written: "
        f"{len(samples)} samples, rag={corrector is not None}"
    )


if __name__ == "__main__":
    main()
