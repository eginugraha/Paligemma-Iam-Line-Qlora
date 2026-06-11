"""Central configuration for SP-5 (history persistence + object storage).

Reuses the SP-3 Postgres DSN (HTR_PG_DSN) so both sub-projects share one database, and adds
the MinIO object-store settings used to persist end-user uploaded images.

Design decision — module-level constants:
    All values are read at *import time* (module top-level), not inside functions. This makes
    the config a plain dict-like namespace: callers just read `config.PG_DSN`, no call needed.
    The test suite exploits this by calling importlib.reload(cfg) after monkeypatching env vars,
    which re-executes all module-level statements and re-reads the patched environment.

Design decision — shared database:
    HTR_PG_DSN is the *same* env var SP-3 already defines. Both sub-projects write to different
    tables in the same Postgres instance, keeping the deployment footprint minimal (one DB, one
    connection pool) while keeping the Python packages separate.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Optional: load credentials from a repo-root .env file.
#
# load_dotenv does NOT override real shell exports, so CI/production values
# (set in the shell) always win over the developer's .env file. The try/except
# makes python-dotenv optional: if it isn't installed the module still works,
# it just falls back to whatever is already in the environment.
#
# Path resolution: __file__ is  src/htr_sp5/config.py
#   .parents[0]  = src/htr_sp5/
#   .parents[1]  = src/
#   .parents[2]  = <repo root>   ← where .env lives
# ---------------------------------------------------------------------------
try:
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # parents[2] == repo root
except ImportError:
    pass  # python-dotenv not installed; rely on the shell environment instead

# --- Postgres (shared with SP-3) --------------------------------------------------------

# Connection string for the PostgreSQL database. Shared with SP-3 so eval results and vocab
# vectors live in the same DB instance. The default points to a local dev DB; production
# credentials are injected via the HTR_PG_DSN environment variable (or the .env file above).
PG_DSN: str = os.environ.get("HTR_PG_DSN", "postgresql://localhost:5432/htr")

# Table that holds one row per evaluation job (a "run" groups many per-image results).
# Named "eval_run" rather than "batch" to be self-documenting in SQL queries.
EVAL_RUN_TABLE: str = "eval_run"

# Table that holds one row per (run, image, scenario) triple — the individual HTR outputs
# with their CER/WER scores. Foreign-keyed to eval_run in the schema.
EVAL_RESULT_TABLE: str = "eval_result"

# Table that holds one row per uploaded image, linking the original filename, the MinIO
# object key, and the upload timestamp. Used by the dashboard's history view.
UPLOAD_TABLE: str = "upload_result"

# --- MinIO object storage (uploaded images) ---------------------------------------------

# MinIO server address, e.g. "localhost:9000" for local dev or "minio.example.com" for prod.
# An empty string means MinIO is not configured; uploads will be skipped (see minio_configured).
MINIO_ENDPOINT: str = os.environ.get("HTR_MINIO_ENDPOINT", "")

# Access key (username) for the MinIO server. Equivalent to AWS_ACCESS_KEY_ID in S3 terms.
MINIO_ACCESS_KEY: str = os.environ.get("HTR_MINIO_ACCESS_KEY", "")

# Secret key (password) for the MinIO server. Equivalent to AWS_SECRET_ACCESS_KEY in S3 terms.
MINIO_SECRET_KEY: str = os.environ.get("HTR_MINIO_SECRET_KEY", "")

# Bucket name where uploaded images are stored. The bucket is created on first use if it does
# not already exist (handled in objectstore.py). Defaults to "htr-uploads".
MINIO_BUCKET: str = os.environ.get("HTR_MINIO_BUCKET", "htr-uploads")

# Whether to connect to MinIO over TLS (HTTPS). Parse the env string to a Python bool:
#   "true"  (case-insensitive) → True   (use HTTPS, required for public deployments)
#   anything else              → False  (use HTTP, fine for local/Docker dev)
MINIO_SECURE: bool = os.environ.get("HTR_MINIO_SECURE", "false").strip().lower() == "true"


def minio_configured() -> bool:
    """Return True only when all three MinIO credentials are non-empty strings.

    Used as a guard in upload code paths: if MinIO is not configured (e.g. in local dev or
    during unit tests), the upload hook silently no-ops rather than raising a connection error.
    All three values must be set because a partial config (e.g. endpoint without keys) would
    cause an authenticated request to fail at runtime with an opaque error.

    Returns:
        bool: True if MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY are all non-empty.
    """
    # bool() treats empty string as False, so this collapses the three truthiness checks into one.
    return bool(MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY)
