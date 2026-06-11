# SP-5 — Batch Evaluation & Dashboard

SP-5 adds two persistent history layers and two new frontend pages that together produce the
**Bab 4 comparison matrix** required by the thesis:

- **Batch eval history** — `scripts/eval_sp5.py` runs M1–M4 over the IAM-test split, stores
  per-sample CER/WER in Postgres, and groups them into named `eval_run` records.
- **Upload history** — images uploaded through the `/v1/detect` endpoint are persisted
  best-effort to MinIO (object storage) and their metadata/results are saved to the same
  Postgres database as SP-3 (`HTR_PG_DSN`).
- **`/dashboard` page** — an FR-FE-05 statistics matrix (mean CER/WER per scenario per run)
  plus a Chart.js grouped bar chart comparing CER and WER across M1–M4.
- **`/history` page** — paginated table of end-user uploads with thumbnails (served via
  presigned URL) and an expandable detail panel showing the per-scenario transcription results.

Images are stored in **MinIO**; all metadata and results live in the **same PostgreSQL
instance** as the SP-3 vocabulary store (`HTR_PG_DSN`).

## Schema setup (one-time)

Run once against your Postgres instance before the first eval or upload:

    python -c "import sys; sys.path.insert(0,'src'); from htr_sp5.store import Sp5Store; Sp5Store().create_schema()"

`create_schema()` uses `CREATE TABLE IF NOT EXISTS`, so it is safe to re-run.

## Run a batch evaluation (needs GPU engine + HTR_PG_DSN)

    # M1/M2 only — 50 IAM-test samples
    python scripts/eval_sp5.py --limit 50

    # M1/M2/M3/M4 — 200 samples with RAG correction (needs HTR_ENABLE_RAG=1 + pgvector ingested)
    python scripts/eval_sp5.py --limit 200 --rag --model-ref <hf-repo>

Available flags:

| flag | default | purpose |
|---|---|---|
| `--limit N` | 200 | number of IAM-test samples to evaluate |
| `--rag` | off | enable M3/M4 via the SP-3 RAG corrector |
| `--model-ref LABEL` | none | version label stored on the run row (e.g. HF repo id) |
| `--dataset LABEL` | `iam-line-test` | human-readable dataset label on the run row |

Each invocation writes one `eval_run` row (with aggregate CER/WER) and one `eval_result` row
per sample. Results are immediately visible on `/dashboard` after the run finishes.

## MinIO — upload persistence

Set the `HTR_MINIO_*` variables (see `.env.example`):

    HTR_MINIO_ENDPOINT=localhost:9000
    HTR_MINIO_ACCESS_KEY=minioadmin
    HTR_MINIO_SECRET_KEY=minioadmin
    HTR_MINIO_BUCKET=htr-uploads
    HTR_MINIO_SECURE=false

When these are set, every image sent to `POST /v1/detect` is stored in MinIO (best-effort —
the detection stream is not blocked if the upload fails). The upload record appears on the
`/history` page with a thumbnail loaded through `GET /v1/uploads/{id}/image` (a presigned
MinIO URL). If the `HTR_MINIO_*` variables are unset, detection and streaming continue to
work normally; uploads just are not persisted.

Easiest local MinIO setup:

    docker run -p 9000:9000 -p 9001:9001 \
      -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
      quay.io/minio/minio server /data --console-address ":9001"

## View dashboard and history

Start the backend with RAG enabled, then start the frontend:

    HTR_ENABLE_RAG=1 uvicorn htr_sp2.api:app --app-dir src
    cd frontend && npm run dev

Then open:
- `http://localhost:5173/dashboard` — batch eval matrix + CER/WER bar chart.
- `http://localhost:5173/history` — upload history with thumbnails and detail panels.

## New API endpoints

| method | path | description |
|---|---|---|
| `GET` | `/v1/eval/runs` | list all eval runs (id, dataset, model_ref, created_at, mean CER/WER) |
| `GET` | `/v1/eval/summary?run_id=` | per-scenario aggregate stats for one run |
| `GET` | `/v1/uploads?limit=&offset=` | paginated list of upload records |
| `GET` | `/v1/uploads/{id}/image` | presigned MinIO URL redirect for the stored image |

## Tests

    python -m pytest -q           # SP-5 DB roundtrip skipped without HTR_PG_TEST
    cd frontend && npm test
    cd frontend && npm run check

## Manual e2e smoke (not automated — needs live MinIO + Postgres + trained engine)

1. Create schema (above) + set `HTR_MINIO_*` and `HTR_PG_DSN` in `.env` or shell.
2. `HTR_ENABLE_RAG=1 uvicorn htr_sp2.api:app --app-dir src` in one terminal;
   `cd frontend && npm run dev` in another.
3. Upload a handwriting PNG on the detect page → confirm it appears on `/history` with a
   working thumbnail and an expandable detail panel showing M1–M4 results.
4. `python scripts/eval_sp5.py --limit 5` → open `/dashboard` → verify the matrix table
   and the CER/WER bar chart both render with data.

## Scope

Batch evaluation storage, upload persistence, `/dashboard`, and `/history` only. M1/M2 engine
lives in SP-2 (`README-sp2.md`); M3/M4 RAG correction and pgvector setup live in SP-3
(`README-sp3.md`); the Svelte frontend shell lives in SP-4 (`README-sp4.md`).
