"""run_eval drives detect_stream over samples and writes rows via an injected store."""
from PIL import Image
from htr_sp5.evalrun import run_eval


class FakeEngine:
    """Minimal InferenceEngine: returns a fixed transcription regardless of input."""
    def run(self, image, prompt, max_new_tokens):
        return "the quick brown fox"


class RecordingStore:
    def __init__(self):
        self.run = None
        self.rows = []

    def create_eval_run(self, dataset, n_samples, model_ref, rag_enabled, notes=None):
        self.run = dict(dataset=dataset, n_samples=n_samples, model_ref=model_ref, rag_enabled=rag_enabled)
        return 1

    def insert_eval_results(self, run_id, rows):
        self.rows.extend(rows)


def _samples(n):
    img = Image.new("RGB", (8, 8), (255, 255, 255))
    return [{"sample_id": f"s{i}", "image": img, "ground_truth": "the quick brown fox"} for i in range(n)]


def test_run_eval_creates_run_and_inserts_rows_per_sample():
    store = RecordingStore()
    run_id = run_eval(_samples(2), FakeEngine(), corrector=None, store=store,
                    dataset="iam-line-test", model_ref="x")
    assert run_id == 1
    assert store.run["n_samples"] == 2 and store.run["rag_enabled"] is False
    # No corrector -> only m1 and m2 per sample -> 2 scenarios x 2 samples = 4 rows.
    assert len(store.rows) == 4
    assert {r.scenario for r in store.rows} == {"m1", "m2"}
    assert all(r.sample_id in {"s0", "s1"} for r in store.rows)
    assert next(r for r in store.rows if r.scenario == "m1").cer == 0.0
