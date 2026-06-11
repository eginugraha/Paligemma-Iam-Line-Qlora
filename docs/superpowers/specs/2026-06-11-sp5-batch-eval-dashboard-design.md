# SP-5 — Batch Evaluation, Dashboard & Upload History — Design

**Date:** 2026-06-11
**Status:** Approved (design)
**Depends on:** SP-2 (`POST /v1/detect`, orchestrator M1–M4, CER/WER), SP-3 (Postgres + pgvector store, `HTR_PG_DSN`), SP-4 (Svelte frontend, navbar host).
**Blocks:** — (final sub-project of the thesis system)

---

## 1. Purpose

Implement the PRD's **FR-FE-05** (global statistics dashboard) plus persistent **history** of
results. The system keeps **two histories**:

1. **Dataset / batch-eval history** — results of running M1–M4 over a sample of the IAM test set,
   aggregated into the summary matrix needed for the thesis **Bab 4** appendix.
2. **End-user upload history** — every image uploaded through the frontend `/detect` flow is
   persisted (image + M1–M4 results) and browsable on a `/history` page.

Both histories live in the **same Postgres database already used by SP-3** (pgvector). Uploaded
images are stored in **MinIO** (S3-compatible object storage), referenced from the DB by object
key.

## 2. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Storage backend | **Reuse the remote Postgres** (same DB as SP-3 pgvector); add 3 tables |
| Image storage | **MinIO** object storage (`minio-py` client); DB stores the **object key** only |
| Batch eval trigger | **Offline CLI** `scripts/eval_sp5.py` (needs GPU + many images; not browser-triggered) |
| Upload persistence | **Automatic** in `POST /v1/detect`, **after** the stream completes |
| Frontend layout | **2 separate routes** — `/dashboard` (batch matrix + chart) and `/history` (upload list) + navbar |
| Dashboard chart | **Chart.js** bar chart (CER & WER per scenario) via a thin `BarChart.svelte` wrapper, alongside the matrix table |
| `eval_result` shape | **Long format** (1 row per image×scenario) → easy `GROUP BY scenario` aggregation |
| `upload_result` shape | Results stored as **JSONB** (display-only, not aggregated) |
| Image reference column | `object_key` (MinIO key string, e.g. `uploads/2026/06/11/<uuid>.png`) |
| Thumbnail delivery | Backend returns a **presigned GET URL** (no FastAPI StaticFiles) |
| DB access | **psycopg v3**, DSN from `HTR_PG_DSN` — same pattern as `htr_sp3/store.py` |
| Testing | pytest (backend, MinIO mocked) + Vitest (frontend, fixtures); no Playwright |

## 3. Architecture & components

New Python package + CLI + frontend routes, mirroring the existing `htr_sp1/2/3` layout:

```
src/htr_sp5/
  config.py        # read HTR_PG_DSN (reuse SP-3) + HTR_MINIO_* settings
  store.py         # psycopg v3: CREATE TABLE + insert/query helpers (htr_sp3/store.py pattern)
  objectstore.py   # minio-py wrapper: put_object + presigned_get_url
  schemas.py       # dataclasses: EvalRun, EvalResult, UploadResult
scripts/
  eval_sp5.py      # offline CLI: run M1-M4 over N IAM-test samples -> eval_run + eval_result
frontend/src/routes/
  +layout.svelte           # navbar: Detect | Dashboard | History
  dashboard/+page.svelte   # FR-FE-05 batch-eval summary matrix + comparison chart
  history/+page.svelte     # upload history list + thumbnail + expandable detail
frontend/src/lib/
  BarChart.svelte          # thin Chart.js wrapper (grouped CER/WER bar chart)
```

Small additions to **SP-2** (`src/htr_sp2/api.py`):
- After the `/v1/detect` stream finishes, upload the image to MinIO and insert an `upload_result`
  row (best-effort; a persistence failure is logged but never fails the user's response).
- New read-only endpoints (see §5).

## 4. Data schema (Postgres)

```sql
-- One row per batch-eval run (created by scripts/eval_sp5.py)
CREATE TABLE eval_run (
  id            BIGSERIAL PRIMARY KEY,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  dataset       TEXT    NOT NULL,        -- e.g. 'iam-line-test'
  n_samples     INT     NOT NULL,
  model_ref     TEXT,                    -- adapter/model version (HTR_HUB_REPO_ID)
  rag_enabled   BOOLEAN NOT NULL,        -- whether M3/M4 were included
  notes         TEXT
);

-- One row per (image x scenario) in a run -- long format for easy aggregation
CREATE TABLE eval_result (
  id               BIGSERIAL PRIMARY KEY,
  run_id           BIGINT NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
  sample_id        TEXT NOT NULL,        -- IAM image id
  scenario         TEXT NOT NULL,        -- 'm1_qlora' | 'm2_cot' | 'm3_rag' | 'm4_hybrid'
  text             TEXT,
  ground_truth     TEXT,
  cer              REAL,                 -- NULL if no ground truth
  wer              REAL,
  latency_seconds  REAL,
  log              TEXT,
  status_tag       TEXT
);
CREATE INDEX ON eval_result (run_id, scenario);

-- One row per end-user upload (auto-inserted by POST /v1/detect)
CREATE TABLE upload_result (
  id            BIGSERIAL PRIMARY KEY,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  filename      TEXT NOT NULL,
  object_key    TEXT NOT NULL,           -- MinIO object key (bucket from env)
  ground_truth  TEXT,                    -- nullable
  results       JSONB NOT NULL           -- {m1_qlora:{text,cer,wer,latency_seconds,log,status_tag}, m2_cot:..., m3_rag:..., m4_hybrid:...}
);
```

Notes:
- `eval_result` is **long format** (4 rows per image) so the dashboard matrix is a single
  `AVG(...) GROUP BY scenario`.
- `upload_result.results` is **JSONB** because upload history is displayed as-is, never
  aggregated; this stays flexible if the scenario set changes.
- CER/WER are nullable, matching the SP-2 contract (absent ground truth → `null`).

## 5. API (additions to `htr_sp2/api.py`)

```
POST /v1/detect                       (existing) -- after stream completes:
                                        put image to MinIO + insert upload_result (best-effort)

GET  /v1/eval/runs                    -> list of eval_run (id, created_at, dataset, n_samples,
                                          model_ref, rag_enabled)
GET  /v1/eval/summary?run_id=<id>     -> per-scenario aggregate for one run (default: latest run):
                                          { scenario, avg_cer, avg_wer, avg_latency_seconds, n }[]
GET  /v1/uploads?limit=&offset=       -> upload history, newest first, paginated
GET  /v1/uploads/{id}/image           -> 302 redirect to a presigned MinIO GET URL
```

Aggregation for `/v1/eval/summary` is computed in SQL (`AVG(cer)`, `AVG(wer)`,
`AVG(latency_seconds)`, `COUNT(*)` `GROUP BY scenario`), not in Python.

## 6. Data flows

**Batch eval (offline):**
```
python scripts/eval_sp5.py --limit 200 [--rag]
  1. store.create_eval_run(dataset, n_samples, model_ref, rag_enabled) -> run_id
  2. for each IAM-test sample:
       run M1..M4 via htr_sp2.orchestrator
       compute CER/WER vs ground_truth via htr_sp1.metrics
       store.insert_eval_result(run_id, sample_id, scenario, ...) x4
  3. done -> dashboard reads via GET /v1/eval/summary
```

**End-user upload (online):**
```
browser POST /v1/detect (image [+ ground_truth])
  -> backend streams M1..M4 to browser (SP-4 behaviour unchanged)
  -> AFTER the stream finishes:
       objectstore.put_object(bucket, object_key, image_bytes)
       store.insert_upload_result(filename, object_key, ground_truth, results_json)
       (failure here is logged, not surfaced to the user)
  -> /history reads via GET /v1/uploads; thumbnails via GET /v1/uploads/{id}/image
```

## 7. Frontend

**Navbar** (`+layout.svelte`): `Detect | Dashboard | History` — plain links, Svelte scoped CSS,
no UI library (consistent with SP-4). Backend base URL via existing `VITE_API_BASE`.

**`/dashboard` — FR-FE-05 global statistics matrix:**
- Run selector (dropdown) populated from `GET /v1/eval/runs`; defaults to the latest run.
- Table: rows = M1 QLoRA / M2 +CoT / M3 +RAG / M4 Hybrid; columns = Avg CER, Avg WER,
  Avg Latency, N. Numbers are copy-able for the thesis appendix.
- **Comparison chart (Chart.js):** a grouped **bar chart** of Avg CER and Avg WER per scenario
  (M1–M4), driven by the same summary JSON as the table. Wrapped in a thin `BarChart.svelte`
  component (mounts a Chart.js instance on a `<canvas>`, destroys it on unmount, updates on data
  change). This is the one intentional exception to SP-4's "no UI library" rule — added for the
  thesis Bab 4 comparison visual. No other chart types.

**`/history` — upload history:**
- Paginated list (newest first) from `GET /v1/uploads`.
- Each row: thumbnail (`/v1/uploads/{id}/image`), filename + timestamp, compact M1–M4
  text/CER summary.
- Click a row to expand a detail panel showing the full per-scenario parameters
  (text, CER, WER, latency, log/reasoning, status_tag) — same five parameters as SP-4.

## 8. Configuration (new env keys)

```
# Reuse from SP-3
HTR_PG_DSN=postgresql://user:pass@host:5432/htr

# New: MinIO object storage for uploaded images
HTR_MINIO_ENDPOINT=localhost:9000
HTR_MINIO_ACCESS_KEY=...
HTR_MINIO_SECRET_KEY=...
HTR_MINIO_BUCKET=htr-uploads
HTR_MINIO_SECURE=false        # true when behind TLS
```

Document these in `.env.example`. If MinIO env is unset, the upload-persistence hook is a no-op
(the `/v1/detect` stream still works) so local dev without MinIO is unaffected.

## 9. Testing

**Backend (pytest):**
- `store.py` insert/query round-trips and the `summary` aggregation (test DB or monkeypatched
  connection).
- New read endpoints via FastAPI `TestClient`.
- The `/v1/detect` upload-persistence hook runs after the stream, with MinIO mocked; verify a
  MinIO/DB failure does not break the streamed response.

**Frontend (Vitest):**
- Dashboard table renders from a fixture summary JSON; run selector switches data.
- `BarChart.svelte` mounts/updates/destroys cleanly from fixture data (Chart.js mocked so the
  test asserts the wrapper's lifecycle + passed datasets, not pixel output).
- History list renders + row expand; navbar renders/links.
- New API client functions.
- No Playwright (consistent with SP-4).

**Manual e2e smoke:**
- `python scripts/eval_sp5.py --limit 5` → confirm `/dashboard` shows the matrix.
- Upload via `/detect` → confirm it appears in `/history` with a working thumbnail.

## 10. Out of scope (YAGNI)

- Triggering batch eval from the browser (CLI only).
- Auth / multi-user separation of histories.
- Editing or deleting history rows from the UI.
- Chart types beyond the single grouped CER/WER bar chart (no line/pie/interactive dashboards).
- Storing batch-eval dataset images in MinIO (they are reproducible from the IAM dataset).
