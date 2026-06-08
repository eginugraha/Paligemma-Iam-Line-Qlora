#!/usr/bin/env python
"""SP-1 training entry point for servers/CLI (e.g. a RunPod A5000 Pod).

This is a *thin* launcher: it puts the repo's `src/` on the import path (so the package works
without `pip install -e .`) and hands off to `htr_sp1.cli.main`, where all the real logic and
tests live. There is intentionally NO Colab/Drive code here — persistence comes from the
machine's own disk (e.g. a RunPod Network Volume mounted at /workspace).

Usage:
    export HTR_OUTPUT_DIR="/workspace/outputs"
    export HTR_HUB_REPO_ID="your-hf-username/paligemma-iam-line-qlora"
    huggingface-cli login
    python scripts/train_sp1.py                 # full pipeline, auto precision
    python scripts/train_sp1.py --skip-sanity   # see `--help` for all flags
    nohup python scripts/train_sp1.py > train.log 2>&1 &   # headless, survives logout
"""
import sys
from pathlib import Path

# src/ sits next to the repo root (one level up from scripts/). Prepend it so `htr_sp1`
# is importable without installing the package — same approach as the test conftest.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htr_sp1.cli import main  # noqa: E402 (import after sys.path setup is intentional)

if __name__ == "__main__":
    main()
