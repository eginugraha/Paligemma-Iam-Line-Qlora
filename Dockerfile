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

# Application code. handler.py (repo root) imports the htr_sp1 (model + inference) and
# htr_sp2 (runpod_io wire format) packages, which live under src/.
COPY src/ ./src/
COPY handler.py ./handler.py

# src/ holds the importable packages (htr_sp1, htr_sp2). Putting it on PYTHONPATH lets
# `from htr_sp1 ...` / `from htr_sp2 ...` resolve without a `pip install -e .` step — the
# same convention the local backend and scripts use via `--app-dir src`.
ENV PYTHONPATH=/app/src

# Point HuggingFace at RunPod's cache path. RunPod's "Cached Models" feature pre-downloads the
# model you list on the endpoint to /runpod-volume/huggingface-cache/hub/ (HF cache layout), so
# HF_HOME must be its PARENT (/runpod-volume/huggingface-cache) for the handler to find the
# cached weights instead of re-downloading. With the base (google/paligemma-3b-pt-448) cached,
# cold start drops from ~90 s+ (6 GB download) to ~15 s (local NVMe load). Requires a network
# volume attached to the endpoint; without one it falls back to the ephemeral layer.
ENV HF_HOME=/runpod-volume/huggingface-cache

# handler.py calls runpod.serverless.start({"handler": handler}) at module level; this is
# the entrypoint RunPod invokes for every job. -u keeps logs unbuffered.
CMD ["python", "-u", "handler.py"]
