"""RunPod Serverless handler — runs on a GPU worker, NOT on the local backend.

Loads the SP-1 fine-tuned model once on cold start, then for each request decodes the
image, runs the supplied prompt through htr_sp1.inference.generate_transcription, and
returns {"text": ...}. The local backend talks to this via RunPodEngine.

Why this file lives at the repository ROOT and is named ``rp_handler.py``:
    RunPod's "deploy from GitHub" pipeline scans the repo ROOT for a Python file containing a
    ``runpod.serverless.start(...)`` call to validate the worker (the source of the
    "Could not find runpod.serverless.start() in your repo" error when the handler is buried
    in a subdirectory). The official runpod-workers/worker-basic example uses exactly this
    name and location, so we mirror it. NOTE: the handler must NOT live in a directory named
    ``runpod/`` — that directory would shadow the installed ``runpod`` SDK as a namespace
    package and break ``import runpod``.

Deploy: the repo-root Dockerfile builds an image from requirements-runpod.txt with this file
as the entrypoint (``python -u rp_handler.py``). The generation path requires a GPU, so it is
validated on RunPod (manual/integration), not in the CPU unit suite. The request/response
wire format is unit-tested via htr_sp2.runpod_io.
"""
from __future__ import annotations

import os

import runpod

from htr_sp1.inference import generate_transcription
from htr_sp2 import runpod_io

# Loaded lazily on first request and cached for the worker's lifetime (avoids reloading
# the 3B model on every call).
_MODEL = None
_PROCESSOR = None


def _load_model():
    """Load the SP-1 base + LoRA adapter once, at the configured precision.

    Reuses `htr_sp1.model.load_eval_model` — the exact, tested loader used for evaluation — so the
    served model matches the reported numbers and the merge/precision logic lives in ONE place.

    The base PaliGemma (`config.BASE_MODEL_ID`, a fixed thesis design constant) is downloaded from
    the Hub and quantized in-memory; no merged model is needed. Precision is set by env
    `HTR_BASE_PRECISION`:
      - "4bit" (default): QLoRA config — lightest VRAM (~4 GB), reproduces the baseline numbers.
                          Requires a CUDA GPU (bitsandbytes); the RunPod worker always has one.
      - "bf16" / "fp32" : full-precision base (heavier; use if you want zero precision compromise).

    `HTR_ADAPTER_ID` is the trained adapter (Hub repo or local path). Imports are local so the
    module stays cheap to import (and CPU-only tooling never needs torch/bitsandbytes).
    """
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from htr_sp1.model import load_eval_model

        adapter_id = os.environ["HTR_ADAPTER_ID"]                 # HF repo or local LoRA adapter
        base_precision = os.environ.get("HTR_BASE_PRECISION", "4bit")
        _MODEL, _PROCESSOR = load_eval_model(adapter_id, base_precision=base_precision)
    return _MODEL, _PROCESSOR


def handler(event: dict) -> dict:
    """RunPod entrypoint: {"input": {image_b64, prompt, max_new_tokens}} -> {"text": ...}."""
    args = runpod_io.parse_input(event)
    model, processor = _load_model()
    text = generate_transcription(
        model,
        processor,
        image=args["image"],
        prompt=args["prompt"],
        max_new_tokens=args["max_new_tokens"],
    )
    return {"text": text}


# RunPod entrypoint — called at MODULE level with NO `if __name__ == "__main__"` guard, to
# exactly match every example in the RunPod handler-functions docs
# (https://docs.runpod.io/serverless/workers/handler-functions). RunPod's deploy-from-GitHub
# scanner looks for this call in the module body; wrapping it in a __main__ guard hides it and
# triggers "Could not find runpod.serverless.start() in your repo". The Dockerfile runs this
# file directly (`python -u rp_handler.py`), so start() fires the same way at container boot.
# This module is never imported by the CPU test suite (it needs torch/runpod on a GPU), so
# running start() on import is safe.
runpod.serverless.start({"handler": handler})
