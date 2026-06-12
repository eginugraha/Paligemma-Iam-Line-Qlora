"""RunPod Serverless handler — runs on a GPU worker.

Loads the SP-1 fine-tuned model once on cold start, then transcribes each request's image
with the given prompt and returns {"text": ...}. The local backend talks to this via
RunPodEngine; the wire format is in htr_sp2.runpod_io. Entry point: `python -u handler.py`
(see the repo-root Dockerfile).
"""
from __future__ import annotations

import os

import runpod

from htr_sp1.inference import generate_transcription
from htr_sp2 import runpod_io

# Cached for the worker's lifetime so the 3B model is loaded only on the first request.
_MODEL = None
_PROCESSOR = None


def _load_model():
    """Load the SP-1 base + LoRA adapter once, at HTR_BASE_PRECISION (default 4bit)."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from htr_sp1.model import load_eval_model

        adapter_id = os.environ["HTR_ADAPTER_ID"]
        base_precision = os.environ.get("HTR_BASE_PRECISION", "4bit")
        _MODEL, _PROCESSOR = load_eval_model(adapter_id, base_precision=base_precision)
    return _MODEL, _PROCESSOR


def handler(event: dict) -> dict:
    """{"input": {image_b64, prompt, max_new_tokens}} -> {"text": ...}."""
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


runpod.serverless.start({"handler": handler})
