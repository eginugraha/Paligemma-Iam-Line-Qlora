"""SP-5 additions to the SP-2 API: upload persistence hook + read endpoints.

We monkeypatch the persistence dependencies so no MinIO/Postgres is needed.

Thesis context (SP-5, Chapter 5 — History & Dashboard)
--------------------------------------------------------
These tests verify the four SP-5 extensions layered onto the existing SP-2 ``/v1/detect``
endpoint, without requiring a live MinIO instance or Postgres database.  The technique
used is monkeypatching: we replace the two lazy-provider helpers (``_get_store`` and
``_get_object_store``) at the *module level* in ``htr_sp2.api`` so the endpoint code
picks up our in-memory fakes instead of the real database/MinIO clients.

Why monkeypatch the provider functions rather than environment variables?
-------------------------------------------------------------------------
* Monkeypatching module-level callables is instantaneous (no import-time reload needed)
  and scoped to the test via pytest's ``monkeypatch`` fixture — it is automatically
  undone after each test.
* Environment-variable patching would require ``importlib.reload(sp5_config)`` and
  re-initialising the MinIO/PG client objects, which is slower and harder to reset.
* The two ``_get_store`` / ``_get_object_store`` functions are specifically designed
  to be the monkeypatch seam: they are module-level names, called *inside* each request
  handler rather than at startup, so replacing them at test time affects every future
  request without touching already-initialised objects.
"""
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import htr_sp2.api as api_module


# ---------------------------------------------------------------------------
# Tiny PNG factory
# ---------------------------------------------------------------------------

def _png():
    """Create the smallest valid RGB PNG (8×8 white square) as raw bytes.

    We use a real PIL-encoded PNG rather than hard-coded bytes so the image-
    validation step inside ``/v1/detect`` (``image.load()``) succeeds.  The
    dimensions are small to keep the tests fast.
    """
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# In-memory fakes — stand-ins for Sp5Store and MinioObjectStore
# ---------------------------------------------------------------------------

class RecordingStore:
    """In-memory substitute for ``htr_sp5.store.Sp5Store``.

    Records every ``insert_upload`` call so tests can assert on it, and exposes
    canned data for the read methods so the GET endpoints return predictable JSON.
    No database connection is opened.
    """

    def __init__(self):
        # Accumulates one dict per ``insert_upload`` call.
        self.uploads: list[dict] = []

        # Canned data returned by ``list_eval_runs``.
        self.runs = [
            {
                "id": 7,
                "created_at": "2026-06-11T00:00:00Z",
                "dataset": "iam-line-test",
                "n_samples": 2,
                "model_ref": "x",
                "rag_enabled": True,
            }
        ]

        # Canned data returned by ``eval_summary``.
        self.summary = [
            {
                "scenario": "m1",
                "avg_cer": 5.0,
                "avg_wer": 10.0,
                "avg_latency_seconds": 0.7,
                "n": 2,
            }
        ]

    def insert_upload(self, filename, object_key, ground_truth, results):
        """Record the call and return a fake upload id (1)."""
        self.uploads.append(
            dict(
                filename=filename,
                object_key=object_key,
                ground_truth=ground_truth,
                results=results,
            )
        )
        return 1

    def list_eval_runs(self):
        """Return the canned eval-run list."""
        return self.runs

    def latest_run_id(self):
        """Return the id of the most recent eval run."""
        return 7

    def eval_summary(self, run_id):
        """Return per-scenario aggregate metrics for the requested run."""
        return self.summary

    def list_uploads(self, limit, offset):
        """Return a page of upload-history rows."""
        return [
            {
                "id": 1,
                "created_at": "2026-06-11T00:00:00Z",
                "filename": "a.png",
                "object_key": "uploads/a.png",
                "ground_truth": None,
                "results": {"m1": {"text": "hi"}},
            }
        ]

    def get_upload_object_key(self, upload_id):
        """Return the MinIO object key for an upload, or None if not found."""
        return "uploads/a.png" if upload_id == 1 else None


class FakeObjectStore:
    """In-memory substitute for ``htr_sp5.objectstore.MinioObjectStore``.

    Returns a deterministic object key so tests can assert on the stored key
    without needing a running MinIO server.
    """

    def new_object_key(self, filename):
        """Generate a deterministic object key (ignores the actual filename)."""
        return "uploads/fixed.png"

    def put_object(self, object_key, data, content_type="application/octet-stream"):
        """No-op: we don't actually store anything in tests."""
        return object_key

    def presigned_get_url(self, object_key, expires_seconds=3600):
        """Return a fake presigned URL so the redirect test has a predictable target."""
        return f"http://minio/{object_key}"


# ---------------------------------------------------------------------------
# Pytest fixture — wires the fakes into the API module for a single test
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """Create a TestClient with persistence dependencies replaced by in-memory fakes.

    ``monkeypatch.setattr`` replaces the module-level ``_get_store`` and
    ``_get_object_store`` callables in ``htr_sp2.api`` for the duration of a
    single test.  The fixture is function-scoped (default) so every test gets a
    fresh ``RecordingStore`` with an empty ``uploads`` list.

    Yields:
        tuple[TestClient, RecordingStore]: The ASGI test client and the recording
        store so individual tests can inspect what was persisted.
    """
    store = RecordingStore()
    objstore = FakeObjectStore()
    monkeypatch.setattr(api_module, "_get_store", lambda: store)
    monkeypatch.setattr(api_module, "_get_object_store", lambda: objstore)
    return TestClient(api_module.app), store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_detect_persists_upload_after_stream(client):
    """After the detect stream completes, the image must be persisted via the store.

    Specifically: ``insert_upload`` must be called exactly once with the object key
    produced by ``FakeObjectStore.new_object_key`` and a ``results`` dict that
    contains at least the ``"m1"`` scenario key (folded from the stream's result events).

    This verifies the core SP-5 feature: every detect call leaves a record in the
    upload-history table so the dashboard's history page can display it.
    """
    c, store = client
    resp = c.post("/v1/detect", files={"file": ("a.png", _png(), "image/png")})
    assert resp.status_code == 200

    # Exactly one upload should have been recorded.
    assert len(store.uploads) == 1

    up = store.uploads[0]
    # The object key must come from the fake object store.
    assert up["object_key"] == "uploads/fixed.png"
    # The results dict must contain at least the m1 scenario from the FakeEngine stream.
    assert "m1" in up["results"]
    # ``fold_results`` must have extracted the ``text`` field from the result event.
    # FakeEngine's default response is "the quick brown fox" (see engines/fake.py),
    # so we assert the exact value to confirm fold_results correctly mapped the event.
    m1_result = up["results"]["m1"]
    assert "text" in m1_result, "fold_results must include the 'text' key for m1"
    assert m1_result["text"] == "the quick brown fox", (
        f"expected FakeEngine default output 'the quick brown fox', got {m1_result['text']!r}"
    )


def test_eval_runs_and_summary_endpoints(client):
    """The GET /v1/eval/runs and /v1/eval/summary endpoints must proxy the store data.

    ``/v1/eval/runs`` lists all batch evaluation runs; ``/v1/eval/summary`` returns
    per-scenario aggregate metrics for the most recent run (when no run_id is given).
    Both endpoints return the canned data from ``RecordingStore`` as JSON.
    """
    c, _ = client
    # GET /v1/eval/runs → list of run dicts; the first must have id == 7.
    runs = c.get("/v1/eval/runs").json()
    assert runs[0]["id"] == 7

    # GET /v1/eval/summary (no run_id → defaults to latest) → list of scenario aggregates.
    summary = c.get("/v1/eval/summary").json()
    assert summary[0]["scenario"] == "m1"


def test_uploads_list_and_image_redirect(client):
    """The GET /v1/uploads list and /v1/uploads/{id}/image redirect must work.

    ``/v1/uploads`` returns a page of upload-history rows; the first row must have
    filename == "a.png" (matching the canned data in ``RecordingStore``).

    ``/v1/uploads/1/image`` must redirect (307) to the presigned URL generated by
    ``FakeObjectStore.presigned_get_url``, which is ``http://minio/uploads/a.png``.
    We use ``follow_redirects=False`` so httpx does not follow the redirect; instead
    we inspect the 307 status and ``location`` header directly.
    """
    c, _ = client
    # List endpoint.
    uploads = c.get("/v1/uploads").json()
    assert uploads[0]["filename"] == "a.png"

    # Image redirect endpoint.  follow_redirects=False → we see the 307 response.
    r = c.get("/v1/uploads/1/image", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "http://minio/uploads/a.png"


def test_persistence_failure_does_not_break_stream(client, monkeypatch):
    """A crash in the persistence layer must NOT affect the HTTP response.

    This is the key safety guarantee of the SP-5 upload hook: it runs *after* the
    stream has been fully yielded to the client, and any exception is caught and
    logged rather than re-raised.  The client must still receive a well-formed
    stream ending with a ``done`` event, regardless of a storage failure.

    We simulate a storage failure by monkeypatching ``insert_upload`` to raise
    ``RuntimeError``.  The test then verifies:
      1. The HTTP status is still 200.
      2. The last event in the stream is ``{"event": "done"}``.
    """
    c, store = client

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "insert_upload", boom)

    resp = c.post("/v1/detect", files={"file": ("a.png", _png(), "image/png")})

    # The HTTP response must succeed despite the storage failure.
    assert resp.status_code == 200

    # Parse every non-empty NDJSON line.
    events = [json.loads(l) for l in resp.text.splitlines() if l.strip()]

    # The stream must end with a ``done`` event.
    assert events[-1]["event"] == "done"
