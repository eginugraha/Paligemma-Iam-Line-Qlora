"""Sp5Store persists eval runs/results and uploads in Postgres.

The roundtrip test hits a REAL database, so it is opt-in via HTR_PG_TEST=1 (same guard as
tests/test_sp3_store.py). Without it, we only check the store constructs from config.
"""
import os
import pytest

from htr_sp5.store import Sp5Store


def test_store_constructs_from_config_without_connecting():
    store = Sp5Store(dsn="postgresql://u:p@h:5432/db")
    assert store._dsn == "postgresql://u:p@h:5432/db"


@pytest.mark.skipif(
    not os.environ.get("HTR_PG_TEST"),
    reason="set HTR_PG_TEST=1 (with a live Postgres and HTR_PG_DSN) to run this test",
)
def test_eval_and_upload_roundtrip():
    from htr_sp5.schemas import EvalResultRow

    store = Sp5Store()
    store.create_schema()
    run_id = store.create_eval_run(dataset="iam-line-test", n_samples=1, model_ref="x", rag_enabled=True)
    store.insert_eval_results(run_id, [
        EvalResultRow("s1", "m1", "the cat", "the cat", 0.0, 0.0, 0.7, "Direct.", "Raw Output"),
    ])
    summary = store.eval_summary(run_id)
    assert summary[0]["scenario"] == "m1" and summary[0]["n"] == 1 and summary[0]["avg_cer"] == 0.0

    # Verify latest_run_id() returns the run we just created — covers the 8th public method.
    assert store.latest_run_id() == run_id

    runs = store.list_eval_runs()
    assert runs[0]["id"] == run_id and runs[0]["n_samples"] == 1

    up_id = store.insert_upload(
        filename="a.png", object_key="uploads/a.png", ground_truth="the cat",
        results={"m1": {"text": "the cat", "cer": 0.0}},
    )
    uploads = store.list_uploads(limit=10, offset=0)
    assert uploads[0]["id"] == up_id and uploads[0]["object_key"] == "uploads/a.png"
    assert store.get_upload_object_key(up_id) == "uploads/a.png"
