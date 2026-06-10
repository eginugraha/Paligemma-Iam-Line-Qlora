"""Runtime configuration for the SP-2 backend.

Values come from environment variables so the same code runs locally (fake engine) or
against RunPod without edits. Prompts/token caps live here so M1 and M2 behaviour is in
one place. CoT prompt is imported from `cot` to avoid duplicating the string.
"""
from __future__ import annotations

import os

# Load a local .env (optional) before reading os.environ, so HTR_ENGINE / HTR_RUNPOD_* /
# HTR_M2_MAX_NEW_TOKENS can live in a gitignored .env at the repo root. load_dotenv does not
# override real shell exports, and python-dotenv is optional (falls back to the shell env).
# (htr_sp1.config also loads .env; calling it twice is harmless and idempotent.)
try:
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # parents[2] == repo root
except ImportError:
    pass

from htr_sp1 import config as sp1config
from htr_sp2 import cot

# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

# Which engine get_engine() builds: "fake" (deterministic, no GPU) or "runpod".
# Defaults to "fake" so tests and a fresh checkout work without any cloud credentials.
ENGINE = os.environ.get("HTR_ENGINE", "fake")

# ---------------------------------------------------------------------------
# RunPod Serverless connection (only needed when ENGINE == "runpod")
# ---------------------------------------------------------------------------

# Endpoint ID from the RunPod dashboard, e.g. "abc123xyz".
RUNPOD_ENDPOINT_ID = os.environ.get("HTR_RUNPOD_ENDPOINT_ID", "")

# API key for authenticating against the RunPod REST API.
RUNPOD_API_KEY = os.environ.get("HTR_RUNPOD_API_KEY", "")

# Generous timeout: RunPod cold starts can take a while and there is no hard 5s limit.
# 180 s covers a typical T4 cold-start (model load ~60-90 s) plus generation time.
RUNPOD_TIMEOUT_SECONDS = float(os.environ.get("HTR_RUNPOD_TIMEOUT", "180"))

# ---------------------------------------------------------------------------
# M1 — baseline transcription (same prompt as SP-1)
# ---------------------------------------------------------------------------

# M1 reuses the SP-1 transcription prompt verbatim; changing it here would diverge the
# SP-1 / SP-2 comparison so treat this constant as fixed for the thesis methodology.
M1_PROMPT = sp1config.TRANSCRIPTION_PROMPT

# IAM handwriting lines are short (typically < 30 tokens); cap from SP-1 is sufficient.
M1_MAX_NEW_TOKENS = sp1config.MAX_TARGET_TOKENS

# Human-readable label shown in the frontend and logged in NDJSON result rows.
M1_STATUS_TAG = "Raw Output"

# ---------------------------------------------------------------------------
# M2 — Chain-of-Thought transcription
# ---------------------------------------------------------------------------

# M2 uses the CoT prompt so the model first reasons about ambiguous strokes before
# committing to a final answer. Imported from cot.py — one source of truth.
M2_PROMPT = cot.COT_PROMPT

# CoT output is longer: reasoning prefix + 'Final:' + answer. 256 tokens is the default;
# override per run if longer reasoning is expected (e.g., longer lines or more ambiguity).
M2_MAX_NEW_TOKENS = int(os.environ.get("HTR_M2_MAX_NEW_TOKENS", "256"))

# Human-readable label for M2 results in the frontend and NDJSON logs.
M2_STATUS_TAG = "Reasoned"

# M3 (RAG correction of M1) and M4 (RAG correction of M2/CoT) status badges shown by the
# frontend table. Kept here with the M1/M2 tags so all scenario labels live in one file.
M3_STATUS_TAG = "Corrected"
M4_STATUS_TAG = "Optimal"

# ---------------------------------------------------------------------------
# RAG correction (M3/M4) and CORS — added for the SP-4 frontend
# ---------------------------------------------------------------------------

# M3/M4 RAG correction is OFF unless explicitly enabled. When on, the API builds a
# PgVectorStore-backed corrector, which needs a live, ingested Postgres+pgvector
# (run scripts/ingest_sp3.py first). Off by default so M1/M2 behaviour is unchanged.
ENABLE_RAG = os.environ.get("HTR_ENABLE_RAG", "0").lower() in ("1", "true", "yes")

# Browser origins allowed to call the API (CORS), comma-separated. Defaults to "*" for the
# prototype frontend; credentials are not used so the wildcard is safe.
CORS_ORIGINS = os.environ.get("HTR_CORS_ORIGINS", "*").split(",")
