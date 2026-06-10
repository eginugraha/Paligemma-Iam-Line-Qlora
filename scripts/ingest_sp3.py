#!/usr/bin/env python
"""SP-3 ingestion CLI: build the IAM-train vocabulary and load it into PostgreSQL/pgvector.

Usage:
    # either export the DSN in your shell...
    export HTR_PG_DSN="postgresql://user:pass@localhost:5432/htr"
    python scripts/ingest_sp3.py

    # ...or put it in a local .env file (see .env.example) and just run:
    python scripts/ingest_sp3.py
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load credentials (e.g. HTR_PG_DSN) from a local .env file BEFORE importing any htr_sp3
# module — config.py reads os.environ at import time, so the .env must populate the
# environment first. python-dotenv is optional: if it is not installed we silently fall back
# to whatever is already exported in the shell (the original behaviour, no .env required).
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(SRC.parent / ".env")  # SRC.parent == repo root, where .env lives
except ImportError:
    pass

from htr_sp1 import data  # noqa: E402
from htr_sp3 import ingest  # noqa: E402
from htr_sp3.store import PgVectorStore  # noqa: E402


def main() -> None:
    print("[SP-3 ingest] loading IAM train split...")
    train = data.load_iam_splits()["train"]

    store = PgVectorStore()
    print("[SP-3 ingest] creating schema (truncates existing rows)...")
    store.create_schema()

    print("[SP-3 ingest] building + loading vocabulary...")
    count = ingest.ingest_vocabulary(train, store)

    print(f"[SP-3 ingest] building HNSW index over {count} words...")
    store.create_index()
    print(f"[SP-3 ingest] done. {count} words ingested.")


if __name__ == "__main__":
    main()
