"""detect_stream gains an optional `corrector`. When present, after M1/M2 it emits m3 (correct
M1 text) and m4 (correct M2 text). When absent, behaviour is unchanged (backward-compatible).
A correction error isolates to its own error event; the stream still finishes with `done`.
"""
import json

from htr_sp2.engine import EngineError
from htr_sp2.orchestrator import detect_stream


class FakeEngine:
    """Returns canned raw outputs per prompt so we control M1/M2 text."""

    def __init__(self, m1="medisal", m2="reasoning... Final: recyrd"):
        self._m1, self._m2 = m1, m2

    def run(self, image, prompt, max_new_tokens):
        # M2's CoT prompt is longer; distinguish by which prompt arrives.
        from htr_sp2 import config
        return self._m2 if prompt == config.M2_PROMPT else self._m1


class FakeCorrector:
    def correct(self, text):
        mapping = {"medisal": "medical", "recyrd": "record"}
        fixed = mapping.get(text.strip(), text)
        log = [] if fixed == text else [{"from": text, "to": fixed, "distance": 0.1}]
        return fixed, log


def _events(gen):
    return [json.loads(line) for line in gen]


def test_without_corrector_only_m1_m2():
    events = _events(detect_stream(FakeEngine(), image=None, filename="x.png", ground_truth=None))
    models = [e["model"] for e in events if e.get("event") == "result"]
    assert models == ["m1", "m2"]


def test_with_corrector_emits_m3_and_m4():
    events = _events(detect_stream(
        FakeEngine(), image=None, filename="x.png", ground_truth=None, corrector=FakeCorrector()
    ))
    results = {e["model"]: e for e in events if e.get("event") == "result"}
    assert set(results) == {"m1", "m2", "m3", "m4"}
    assert results["m3"]["text"] == "medical"   # corrected M1
    assert results["m4"]["text"] == "record"    # corrected M2


def test_m4_skipped_when_m2_fails():
    class M2FailEngine(FakeEngine):
        def run(self, image, prompt, max_new_tokens):
            from htr_sp2 import config
            if prompt == config.M2_PROMPT:
                raise EngineError("boom")
            return self._m1

    events = _events(detect_stream(
        M2FailEngine(), image=None, filename="x.png", ground_truth=None, corrector=FakeCorrector()
    ))
    errors = {e["model"] for e in events if e.get("event") == "error"}
    assert "m2" in errors and "m4" in errors
    assert events[-1]["event"] == "done"
