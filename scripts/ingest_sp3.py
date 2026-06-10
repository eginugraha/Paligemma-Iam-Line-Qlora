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

# HTR_PG_DSN is read from a local .env (if present) automatically — htr_sp3.config calls
# load_dotenv() at import time, so no explicit loading is needed here.
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
