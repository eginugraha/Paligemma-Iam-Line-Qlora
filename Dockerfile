# RunPod Serverless worker image for the SP-1/SP-2 HTR inference engine.
#
# This container is the "engine (GPU)" layer in the architecture: the local FastAPI backend
# (htr_sp2) and the eval CLI (scripts/eval_sp5.py) call it over HTTP via RunPodEngine. It
# loads the fine-tuned PaliGemma (base + LoRA adapter) ONCE on cold start and then serves
# transcription requests through runpod/handler.py.
#
# Build path: RunPod's "build from GitHub" pipeline builds this Dockerfile on RunPod's own
# infrastructure — nothing is built or pushed from a local machine. The repo root is the
# build context, so the COPY paths below are relative to the repository root.
#
# Base image: PyTorch 2.3.1 + CUDA 12.1 runtime. This matches the torch pin in
# requirements-runpod.txt and provides the CUDA runtime that bitsandbytes 0.43.1 needs for
# 4-bit (QLoRA) quantization. torch is already present in the base, so the pip install below
# simply finds it satisfied and skips re-downloading it.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Non-interactive apt + unbuffered Python so RunPod's log viewer streams output line-by-line.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install worker-side Python deps in their own layer FIRST so this layer stays cached when
# only application code (src/, runpod/) changes — speeds up subsequent rebuilds.
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir -r requirements-runpod.txt

# Application code. handler.py imports the htr_sp1 (model + inference) and htr_sp2
# (runpod_io wire format) packages, which live under src/.
COPY src/ ./src/
COPY runpod/ ./runpod/

# src/ holds the importable packages (htr_sp1, htr_sp2). Putting it on PYTHONPATH lets
# `from htr_sp1 ...` / `from htr_sp2 ...` resolve without a `pip install -e .` step — the
# same convention the local backend and scripts use via `--app-dir src`.
ENV PYTHONPATH=/app/src

# Cache HuggingFace downloads on the RunPod network volume (mounted at /runpod-volume) so the
# ~6 GB gated base model (google/paligemma-3b-pt-448) and the LoRA adapter are downloaded
# once and reused across cold starts instead of every time the worker scales from zero.
# ATTACH A NETWORK VOLUME to the endpoint for this to persist; without one it falls back to
# the container's ephemeral layer and re-downloads on each cold start.
ENV HF_HOME=/runpod-volume/huggingface

# The handler's __main__ block calls runpod.serverless.start({"handler": handler}); this is
# the entrypoint RunPod invokes for every job. -u keeps logs unbuffered.
CMD ["python", "-u", "runpod/handler.py"]
