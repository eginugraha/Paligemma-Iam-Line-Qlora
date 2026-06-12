# SP-2 — Backend Core

FastAPI backend comparing two HTR scenarios on one handwriting-line image:

- **M1 (Baseline QLoRA):** direct transcription via the SP-1 fine-tuned model.
- **M2 (QLoRA + CoT):** same model, a CoT prompt; reasoning is parsed out of the output.

Results stream back as **NDJSON** (one event per line): `meta`, a `result` (or `error`)
per scenario, then `done`. CER/WER are computed (reusing `htr_sp1.metrics`) only when a
`ground_truth` form field is supplied; otherwise they are `null`.

When an SP-3 corrector is passed to `detect_stream`, two more scenarios stream after M1/M2:
**M3** (RAG-corrected M1) and **M4** (RAG-corrected M2). Without a corrector the stream is
unchanged. See `README-sp3.md`.

## Run locally (fake engine — no GPU)

    pip install -r requirements-backend.txt
    HTR_ENGINE=fake uvicorn htr_sp2.api:app --reload --app-dir src

    curl -N -F file=@line_01.png -F ground_truth="the quick brown fox" \
      http://127.0.0.1:8000/v1/detect

## Run against RunPod

Set the engine + credentials, then start the server:

    export HTR_ENGINE=runpod
    export HTR_RUNPOD_ENDPOINT_ID=...    # RunPod Serverless endpoint id
    export HTR_RUNPOD_API_KEY=...
    uvicorn htr_sp2.api:app --app-dir src

These vars (and the worker's `HTR_ADAPTER_ID` / `HTR_BASE_PRECISION`) can instead live in a
gitignored `.env` at the repo root — see `.env.example`.

## Deploy the RunPod worker

`handler.py` (repo root) is the Serverless entrypoint — kept at the root with a module-level
`runpod.serverless.start(...)` (no `__main__` guard), and NOT inside a `runpod/` directory
(which would shadow the `runpod` SDK), so RunPod's deploy-from-GitHub scanner can find the
`runpod.serverless.start(...)` call. Build the root `Dockerfile` from
`requirements-runpod.txt` and set `HTR_ADAPTER_ID` (the trained LoRA adapter repo/path). The
base model is the fixed `htr_sp1.config.BASE_MODEL_ID`,
loaded at the precision in `HTR_BASE_PRECISION` (default `4bit` — lightest VRAM, reproduces the
baseline; `bf16`/`fp32` also available). The wire format is in `htr_sp2.runpod_io`.

## Tests

    pytest -q

All backend tests run on CPU (fake engine + mocked HTTP); the GPU generation path is
validated on RunPod.

## Scope

M1 + M2 only. M3/M4 (RAG/pgvector) live in SP-3 (`README-sp3.md`) and plug into this
backend via `detect_stream`'s optional `corrector`; the local GGUF engine is a separate
sub-project.
