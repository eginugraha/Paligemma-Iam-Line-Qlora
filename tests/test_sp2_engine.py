"""The engine boundary is 'dumb': run(image, prompt, max_new_tokens) -> raw string.
FakeEngine makes the whole backend testable without a GPU; it records calls and can return
scripted outputs and raise EngineError on chosen call indices (to test error handling)."""
import pytest

from htr_sp2 import engine
from htr_sp2.engines.fake import FakeEngine


def test_fake_engine_returns_scripted_outputs_in_order():
    eng = FakeEngine(responses=["m1 out", "m2 out"])
    assert eng.run(image=object(), prompt="p1", max_new_tokens=64) == "m1 out"
    assert eng.run(image=object(), prompt="p2", max_new_tokens=256) == "m2 out"


def test_fake_engine_records_calls():
    eng = FakeEngine(responses=["a"])
    img = object()
    eng.run(image=img, prompt="p1", max_new_tokens=64)
    assert eng.calls[0]["image"] is img
    assert eng.calls[0]["prompt"] == "p1"
    assert eng.calls[0]["max_new_tokens"] == 64


def test_fake_engine_reuses_last_response_when_exhausted():
    eng = FakeEngine(responses=["only"])
    assert eng.run(image=object(), prompt="p", max_new_tokens=1) == "only"
    assert eng.run(image=object(), prompt="p", max_new_tokens=1) == "only"


def test_fake_engine_raises_on_configured_call_index():
    eng = FakeEngine(responses=["ok", "ok"], fail_on={1})
    eng.run(image=object(), prompt="p", max_new_tokens=1)              # call 0 ok
    with pytest.raises(engine.EngineError):
        eng.run(image=object(), prompt="p", max_new_tokens=1)          # call 1 raises


def test_get_engine_returns_fake_by_default(monkeypatch):
    from htr_sp2 import config
    monkeypatch.setattr(config, "ENGINE", "fake")
    assert isinstance(engine.get_engine(), FakeEngine)


def test_get_engine_unknown_name_raises(monkeypatch):
    from htr_sp2 import config
    monkeypatch.setattr(config, "ENGINE", "nope")
    with pytest.raises(ValueError):
        engine.get_engine()
