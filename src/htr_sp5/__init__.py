"""SP-5: batch evaluation, statistics dashboard, and upload history.

This sub-project adds three capabilities on top of the existing HTR inference pipeline:

1. **Batch evaluation** — run all four HTR scenarios (SP-1 baseline, SP-2 fine-tuned, SP-3 RAG,
   SP-4 CoT) over an arbitrary set of uploaded images in a single job, storing every result row
   in Postgres so the results survive process restarts.

2. **Statistics dashboard** — a Svelte page that reads the persisted eval_run / eval_result rows
   and renders per-scenario CER/WER metrics, comparison charts, and historical run logs.

3. **Upload history** — every image the user uploads through the SP-4 frontend is also stored in
   MinIO (S3-compatible object storage) with a reference row in Postgres, so the user can
   re-run evaluations without re-uploading.

Module layout (populated by later tasks):
    config.py       — env-driven settings (shared PG DSN, MinIO credentials)
    schemas.py      — SQLAlchemy / dataclass schema definitions for the DB tables
    store.py        — Postgres persistence layer (eval_run, eval_result, upload_result)
    objectstore.py  — MinIO upload/download helpers
    evalrun.py      — orchestrator that drives the four scenarios and stores results
"""
