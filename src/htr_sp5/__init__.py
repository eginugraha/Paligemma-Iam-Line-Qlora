"""SP-5: batch evaluation, statistics dashboard, and upload history.

This sub-project adds three capabilities on top of the existing HTR inference pipeline:

1. **Batch evaluation** — run the four HTR scenarios (M1 baseline QLoRA, M2 +CoT, M3 +RAG,
   M4 hybrid) over a sample of the IAM test set in a single offline job, storing every
   per-(sample x scenario) result row in Postgres so the thesis Bab 4 statistics survive
   process restarts.

2. **Statistics dashboard** — a Svelte page (FR-FE-05) that reads the persisted eval_run /
   eval_result rows and renders a per-scenario CER/WER/latency matrix plus a Chart.js
   comparison bar chart.

3. **Upload history** — every image the user uploads through the SP-4 frontend is stored in
   MinIO (S3-compatible object storage) with a reference row in Postgres (image object key +
   the M1-M4 results), so past uploads can be browsed (with thumbnails) on a history page.

Both histories live in the SAME Postgres database used by SP-3's pgvector store (HTR_PG_DSN).

Module layout:
    config.py       — env-driven settings (shared PG DSN, MinIO credentials)
    schemas.py      — plain dataclasses + fold_results() (NDJSON events -> persistable rows)
    store.py        — Postgres persistence layer (eval_run, eval_result, upload_result)
    objectstore.py  — MinIO upload + presigned-URL helpers
    evalrun.py      — run_eval(): drives detect_stream over samples and stores results
"""
