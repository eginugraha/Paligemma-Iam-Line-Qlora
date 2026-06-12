# Handwritten Text Recognition — 4-Scenario Comparison (PaliGemma + CoT + RAG)

> Thesis system that fine-tunes **Google PaliGemma-3B** with **QLoRA** on the **IAM**
> handwriting dataset and compares four transcription strategies (baseline, Chain-of-Thought,
> and two RAG-corrected variants) on a single line image — end to end, from GPU inference to a
> Svelte web UI and an evaluation dashboard.

---

## What is this?

This repository is the complete implementation for an undergraduate thesis on **handwritten
text recognition (HTR)**. The research question: *does adding Chain-of-Thought prompting and
retrieval-augmented (RAG) lexical correction improve a fine-tuned vision-language model's
transcription accuracy?*

To answer it, every uploaded handwriting line is run through **four scenarios** and scored with
**CER / WER**:

| Scenario | Name | What it does |
|---|---|---|
| **M1** | Baseline (QLoRA) | Direct transcription with the fine-tuned PaliGemma |
| **M2** | QLoRA + CoT | Same model, a Chain-of-Thought prompt; reasoning is parsed out |
| **M3** | RAG-corrected M1 | M1 output passed through a pgvector lexical corrector |
| **M4** | RAG-corrected M2 | M2 output passed through the same corrector |

- **Base model:** `google/paligemma-3b-pt-448` (gated)
- **Fine-tuned adapter:** [`eginugraha/paligemma-iam-line-qlora-adapter`](https://huggingface.co/eginugraha/paligemma-iam-line-qlora-adapter)
- **Dataset:** IAM handwriting (line level)

### Baseline result (M1, full IAM-test, 2,915 lines)

| Metric | Value |
|---|---|
| **CER** | **17.37%** |
| **WER** | **28.34%** |
| Perfect transcriptions (CER = 0) | 21.2% |

See [`docs/sp1-initial-eval-2026-06-13.md`](docs/sp1-initial-eval-2026-06-13.md) for the full
distribution, and [`docs/sp3-rag-correction-investigation-2026-06-13.md`](docs/sp3-rag-correction-investigation-2026-06-13.md)
for the ongoing analysis of the RAG (M3/M4) scenarios.

---

## Architecture

```
┌──────────────┐   HTTP    ┌────────────────────────┐   HTTP    ┌─────────────────────┐
│   FRONTEND   │ ────────► │   BACKEND API          │ ────────► │   ENGINE (GPU)      │
│  SvelteKit   │           │   FastAPI (htr_sp2)    │           │   RunPod Serverless │
│  (port 5173) │ ◄──────── │   (port 8000)          │ ◄──────── │   PaliGemma + LoRA  │
└──────────────┘   NDJSON  └────────────────────────┘           └─────────────────────┘
  upload image               orchestrates M1–M4,                  loads model, runs
  show results               RAG correction, CER/WER,             real inference
                             persistence, streaming
                                   │
                         ┌─────────┴──────────┐
                         ▼                    ▼
                   PostgreSQL+pgvector     MinIO (S3)
                   RAG vocab, eval &       uploaded
                   upload history          images
```

The backend is a thin **orchestrator** — it does **not** load the model. Real inference runs on
a **RunPod Serverless GPU worker** (`handler.py` + `Dockerfile`). For local development without
a GPU, set `HTR_ENGINE=fake` to use a deterministic stub engine.

---

## Quickstart

### Prerequisites

- **Python 3.10+** and **Node.js + npm** (frontend)
- For **real inference**: a deployed RunPod Serverless endpoint + credentials in `.env`
  (or use `HTR_ENGINE=fake` for a no-GPU dummy engine)
- *(Optional)* **PostgreSQL + pgvector** for M3/M4 RAG and the eval/history features
- *(Optional)* **MinIO** for persisting uploaded images

### 1. Clone & enter the project

```bash
git clone git@github.com:eginugraha/Paligemma-Iam-Line-Qlora.git
cd Paligemma-Iam-Line-Qlora
```

### 2. Configure environment

```bash
cp .env.example .env
# then edit .env — set HTR_ENGINE, HTR_RUNPOD_*, HTR_PG_DSN, HTR_MINIO_*, HF_TOKEN
```

### 3. Run the backend

Install the (lean, no-torch) backend deps once, then start the API:

```bash
pip install -r requirements-backend.txt

# Start the backend (M1–M4 enabled; engine comes from .env, e.g. runpod or fake)
cd "/Users/eginugraha/personal/Handwritten Text Recognition"
HTR_ENABLE_RAG=1 uvicorn htr_sp2.api:app --app-dir src
```

The API is now at **http://localhost:8000** (Swagger UI at `/docs`).

> `HTR_ENABLE_RAG=1` turns on the M3/M4 RAG scenarios. Drop it for M1/M2 only.
> The engine (`runpod` for real GPU, `fake` for a no-GPU stub) is read from `.env`.

### 4. Run the frontend

In a **second terminal**:

```bash
cd frontend && npm run dev
```

Open **http://localhost:5173** — upload a handwriting line image and the four scenarios stream
in side by side. Also available:

- **http://localhost:5173/dashboard** — evaluation matrix + CER/WER chart
- **http://localhost:5173/history** — upload history with thumbnails

> The frontend and backend are **separate processes** — running the frontend does **not** start
> the backend. Start both (steps 3 and 4) for the app to work.

---

## Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `HTR_ENGINE` | `runpod` (real GPU) or `fake` (no-GPU stub) |
| `HTR_RUNPOD_ENDPOINT_ID`, `HTR_RUNPOD_API_KEY` | RunPod Serverless endpoint credentials |
| `HTR_ENABLE_RAG` | `1` to enable M3/M4 RAG correction |
| `HTR_PG_DSN` | PostgreSQL + pgvector connection string (RAG, eval, history) |
| `HTR_MINIO_*` | MinIO object storage for uploaded images |
| `HF_TOKEN` | HuggingFace token (gated base model + adapter download) |

See `.env.example` for the full annotated list.

---

## Sub-projects

The work is split into five sub-projects, each with its own README:

| | Sub-project | README |
|---|---|---|
| **SP-1** | QLoRA training + evaluation of PaliGemma on IAM | [`README-sp1.md`](README-sp1.md) |
| **SP-2** | Backend core (FastAPI, M1/M2, RunPod engine) | [`README-sp2.md`](README-sp2.md) |
| **SP-3** | RAG correction via pgvector (M3/M4) | [`README-sp3.md`](README-sp3.md) |
| **SP-4** | SvelteKit frontend (detect / compare UI) | (in `frontend/`) |
| **SP-5** | Batch evaluation, dashboard & upload history | [`README-sp5.md`](README-sp5.md) |

---

## Repository structure

```
src/
  htr_sp1/        training, model loading, inference, metrics
  htr_sp2/        FastAPI backend, orchestrator, engines (fake / runpod)
  htr_sp3/        RAG corrector, pgvector store, vocab, threshold tuning
  htr_sp5/        eval & upload persistence (Postgres + MinIO)
frontend/         SvelteKit app (detect, dashboard, history)
scripts/          CLIs: train_sp1, eval_sp1, ingest_sp3, tune_sp3, eval_sp5
handler.py        RunPod Serverless worker entrypoint
Dockerfile        RunPod GPU worker image
reports/          post-training eval metrics + train log
docs/             design specs, plans, and analysis reports
tests/            pytest suite (CPU-only, fake engine + mocks)
```

---

## Tests

```bash
# Backend (CPU-only: fake engine + mocked HTTP/DB)
python -m pytest

# Frontend
cd frontend && npm test && npm run check
```

---

## Deploying the GPU engine (RunPod Serverless)

Real M1–M4 inference runs on a RunPod GPU worker built from the repo-root `Dockerfile` +
`handler.py`. Deploy via RunPod's **build-from-GitHub**, set the worker env
(`HTR_ADAPTER_ID`, `HTR_BASE_PRECISION=4bit`, `HF_TOKEN`), then put the resulting
`HTR_RUNPOD_ENDPOINT_ID` / `HTR_RUNPOD_API_KEY` in your local `.env`. See
[`README-sp2.md`](README-sp2.md) for details.

---

## Contributors

- Putu Bagus Indra Dermawan Kemuning
- Lawy Xenna L. Gaol
- Egi Nugraha

## Citation

If you use this work, please cite the thesis (details TBD) and the underlying components:
PaliGemma (Google), the IAM Handwriting Database, and the QLoRA method.
