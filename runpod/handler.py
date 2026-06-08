"""RunPod Serverless handler — runs on a GPU worker, NOT on the local backend.

Loads the SP-1 fine-tuned model once on cold start, then for each request decodes the
image, runs the supplied prompt through htr_sp1.inference.generate_transcription, and
returns {"text": ...}. The local backend talks to this via RunPodEngine.

Deploy: build an image from requirements-runpod.txt with this as the entrypoint. The
generation path requires a GPU, so it is validated on RunPod (manual/integration), not in
the CPU unit suite. The request/response wire format is unit-tested via htr_sp2.runpod_io.
"""
from __future__ import annotations

import os

import runpod

from htr_sp1.inference import generate_transcription
from htr_sp2 import runpod_io

# Loaded lazily on first request and cached for the worker's lifetime (avoids reloading
# the 3B model on every call). Replace the loader with the SP-1 model assembly used in
# training/export; env vars point at the base model + adapter.
_MODEL = None
_PROCESSOR = None


def _load_model():
    """Load base PaliGemma + LoRA adapter once. Imports are local so the module imports
    cheaply (and so CPU-only tooling never needs torch)."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from peft import PeftModel
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        base_id = os.environ["HTR_BASE_MODEL_ID"]      # e.g. google/paligemma-3b-pt-448
        adapter_id = os.environ["HTR_ADAPTER_ID"]      # HF repo or local path to LoRA adapter
        processor = PaliGemmaProcessor.from_pretrained(base_id)
        base = PaliGemmaForConditionalGeneration.from_pretrained(base_id, device_map="auto")
        _MODEL = PeftModel.from_pretrained(base, adapter_id)
        _PROCESSOR = processor
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


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
