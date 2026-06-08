# SP-2 Backend Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend that takes one handwriting-line image and streams comparative M1 (baseline QLoRA) and M2 (QLoRA + CoT) transcription results as NDJSON, with optional CER/WER, behind a swappable inference engine (RunPod GPU + a deterministic fake).

**Architecture:** New `htr_sp2` package. A "dumb" engine interface returns the model's raw decoded string for `(image, prompt, max_new_tokens)`; all prompt selection (M1 vs M2) and CoT parsing live in the backend so they are testable on CPU. The orchestrator runs M1 then M2, computes metrics by reusing `htr_sp1.metrics`, and yields NDJSON lines. A RunPod Serverless handler (server-side) and `RunPodEngine` (client) share pure (de)serialization in `htr_sp2.runpod_io`. A `FakeEngine` makes the whole backend testable without a GPU.

**Tech Stack:** Python, FastAPI, uvicorn, httpx, Pillow, jiwer (via `htr_sp1`), pytest, pytest-httpx. Server-side handler also uses `runpod` + the SP-1 transformers/peft stack.

---

## File Structure

Created in this plan:

- `requirements-backend.txt` — SP-2 backend runtime + dev deps (kept separate from the SP-1 training stack in `requirements.txt`).
- `requirements-runpod.txt` — server-side handler deps (`runpod` + SP-1 model stack).
- `src/htr_sp2/__init__.py` — package marker.
- `src/htr_sp2/config.py` — engine selector, RunPod settings, prompts, token caps, status tags.
- `src/htr_sp2/cot.py` — `COT_PROMPT` and `parse_cot()` (marker parsing + fallback).
- `src/htr_sp2/runpod_io.py` — pure image encode/decode + request/response (de)serialization, shared by client and handler.
- `src/htr_sp2/schemas.py` — NDJSON event builder functions.
- `src/htr_sp2/engine.py` — `InferenceEngine` Protocol, `EngineError`, `get_engine()` factory.
- `src/htr_sp2/engines/__init__.py`, `src/htr_sp2/engines/fake.py`, `src/htr_sp2/engines/runpod.py`.
- `src/htr_sp2/orchestrator.py` — `detect_stream()` NDJSON generator.
- `src/htr_sp2/api.py` — FastAPI app, `POST /v1/detect`, `GET /health`.
- `runpod/handler.py` — thin server-side handler (model load + glue); generation path is manual/integration.
- `tests/test_sp2_cot.py`, `tests/test_sp2_runpod_io.py`, `tests/test_sp2_schemas.py`, `tests/test_sp2_engine.py`, `tests/test_sp2_orchestrator.py`, `tests/test_sp2_api.py`, `tests/test_sp2_runpod_engine.py`.
- `README-sp2.md` — run/deploy notes.

Modified:

- `src/htr_sp1/inference.py` — add optional `prompt` parameter (backward-compatible).

---

## Task 1: Backend dependencies

**Files:**
- Create: `requirements-backend.txt`
- Create: `requirements-runpod.txt`

- [ ] **Step 1: Write `requirements-backend.txt`**

```
# SP-2 backend (local FastAPI). Separate from requirements.txt (SP-1 training stack)
# so the laptop backend env stays lean — no torch/transformers needed locally.
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
httpx==0.27.0
pillow==10.4.0
jiwer==3.0.4
# Dev / test only:
pytest==8.2.2
pytest-httpx==0.30.0
```

- [ ] **Step 2: Write `requirements-runpod.txt`**

```
# Server-side RunPod Serverless handler deps. Installed in the RunPod worker image,
# NOT on the local backend. The SP-1 model stack does the actual generation.
runpod==1.6.2
torch==2.3.1
transformers==4.42.4
peft==0.11.1
accelerate==0.31.0
pillow==10.4.0
```

- [ ] **Step 3: Install the backend deps**

Run: `pip install -r requirements-backend.txt`
Expected: installs FastAPI, httpx, pytest-httpx, etc. (no torch).

- [ ] **Step 4: Verify pytest still collects the existing suite**

Run: `pytest -q`
Expected: existing SP-1 tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements-backend.txt requirements-runpod.txt
git commit -m "build: add SP-2 backend and RunPod requirement files"
```

---

## Task 2: SP-1 — add optional `prompt` to `generate_transcription`

The handler must drive both the M1 prompt and the M2 CoT prompt through one function. Add an optional `prompt` parameter; default keeps existing behaviour.

**Files:**
- Modify: `src/htr_sp1/inference.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_inference.py`)

```python
def test_generate_transcription_accepts_custom_prompt(fake_model, fake_processor):
    # SP-2 M2 (CoT) drives a different prompt through the same function.
    custom = "describe the strokes then give Final:\n"
    inference.generate_transcription(fake_model, fake_processor, image=object(), prompt=custom)
    assert fake_processor.last_call["text"] == custom
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py::test_generate_transcription_accepts_custom_prompt -v`
Expected: FAIL — `generate_transcription() got an unexpected keyword argument 'prompt'`.

- [ ] **Step 3: Add the parameter**

In `src/htr_sp1/inference.py`, change the signature and the processor call.

Signature (line ~12):
```python
def generate_transcription(model, processor, image, prompt: str = config.TRANSCRIPTION_PROMPT, max_new_tokens: int = config.MAX_TARGET_TOKENS) -> str:
```

In the docstring Args, add:
```python
        prompt: The instruction text. Defaults to the M1 transcription prompt; SP-2 passes
            the CoT prompt for M2. Kept as a parameter so one function serves both modes.
```

In the `processor(...)` call, replace `text=config.TRANSCRIPTION_PROMPT` with:
```python
        text=prompt,
```

- [ ] **Step 4: Run the full inference test file**

Run: `pytest tests/test_inference.py -v`
Expected: all PASS (existing default-prompt test and the new custom-prompt test).

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/inference.py tests/test_inference.py
git commit -m "feat(sp1): generate_transcription accepts optional prompt for SP-2 M2"
```

---

## Task 3: `htr_sp2` package + config

**Files:**
- Create: `src/htr_sp2/__init__.py`
- Create: `src/htr_sp2/config.py`
- Test: `tests/test_sp2_config.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_config.py`)

```python
"""Config holds the knobs the backend reads at runtime. We assert the defaults and the
two engine names so a typo can't silently change behaviour."""
from htr_sp2 import config


def test_default_engine_is_fake():
    # Default to the GPU-free fake so tests and a fresh checkout work without RunPod creds.
    assert config.ENGINE == "fake"


def test_model_prompts_and_tags():
    # M1 reuses the SP-1 transcription prompt; M2 uses the CoT prompt.
    from htr_sp1 import config as sp1config
    from htr_sp2 import cot
    assert config.M1_PROMPT == sp1config.TRANSCRIPTION_PROMPT
    assert config.M2_PROMPT == cot.COT_PROMPT
    assert config.M1_STATUS_TAG == "Raw Output"
    assert config.M2_STATUS_TAG == "Reasoned"
    assert config.M2_MAX_NEW_TOKENS > config.M1_MAX_NEW_TOKENS  # CoT needs room to reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2'`.

- [ ] **Step 3: Create the package marker** (`src/htr_sp2/__init__.py`)

```python
"""SP-2 backend: FastAPI service comparing M1 (baseline) and M2 (CoT) HTR scenarios."""
```

- [ ] **Step 4: Create `src/htr_sp2/config.py`**

```python
"""Runtime configuration for the SP-2 backend.

Values come from environment variables so the same code runs locally (fake engine) or
against RunPod without edits. Prompts/token caps live here so M1 and M2 behaviour is in
one place. CoT prompt is imported from `cot` to avoid duplicating the string.
"""
from __future__ import annotations

import os

from htr_sp1 import config as sp1config
from htr_sp2 import cot

# Which engine get_engine() builds: "fake" (deterministic, no GPU) or "runpod".
ENGINE = os.environ.get("HTR_ENGINE", "fake")

# RunPod Serverless connection (only needed when ENGINE == "runpod").
RUNPOD_ENDPOINT_ID = os.environ.get("HTR_RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.environ.get("HTR_RUNPOD_API_KEY", "")
# Generous timeout: RunPod cold starts can take a while and there is no hard 5s limit.
RUNPOD_TIMEOUT_SECONDS = float(os.environ.get("HTR_RUNPOD_TIMEOUT", "180"))

# M1 reuses the SP-1 transcription prompt and its short token cap (IAM lines are short).
M1_PROMPT = sp1config.TRANSCRIPTION_PROMPT
M1_MAX_NEW_TOKENS = sp1config.MAX_TARGET_TOKENS
M1_STATUS_TAG = "Raw Output"

# M2 uses the CoT prompt and a larger cap so the reasoning + final answer fit.
M2_PROMPT = cot.COT_PROMPT
M2_MAX_NEW_TOKENS = int(os.environ.get("HTR_M2_MAX_NEW_TOKENS", "256"))
M2_STATUS_TAG = "Reasoned"
```

- [ ] **Step 5: Run test**

Run: `pytest tests/test_sp2_config.py -v`
Expected: PASS (requires Task 4's `cot.py`; if running strictly in order, complete Task 4 first then re-run — see note). 

> Note: `config.py` imports `htr_sp2.cot`. Implement Task 4 before running this test. If executing tasks sequentially, write `cot.py` (Task 4) then run Tasks 3 and 4 tests together.

- [ ] **Step 6: Commit** (after Task 4 passes)

```bash
git add src/htr_sp2/__init__.py src/htr_sp2/config.py tests/test_sp2_config.py
git commit -m "feat(sp2): config module (engine selector, prompts, token caps)"
```

---

## Task 4: CoT prompt + parsing

**Files:**
- Create: `src/htr_sp2/cot.py`
- Test: `tests/test_sp2_cot.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_cot.py`)

```python
"""M2 produces reasoning + a final answer in one generation. parse_cot splits them:
text (after the last 'Final:') feeds CER/WER; the reasoning becomes the log. When the
model ignores the format (no marker), we fall back so the comparison stays honest."""
from htr_sp2 import cot


def test_parse_cot_splits_reasoning_and_final():
    raw = "Reasoning: word 4 has a loop 'd'\nFinal: medical"
    text, log = cot.parse_cot(raw)
    assert text == "medical"
    assert log == "Reasoning: word 4 has a loop 'd'"


def test_parse_cot_takes_last_final_marker():
    raw = "Final: draft\nFinal: medical"
    text, log = cot.parse_cot(raw)
    assert text == "medical"
    assert log == "Final: draft"


def test_parse_cot_strips_whitespace():
    text, log = cot.parse_cot("  Final:   hello  ")
    assert text == "hello"
    assert log == ""


def test_parse_cot_fallback_when_no_marker():
    raw = "the quick brown fox"
    text, log = cot.parse_cot(raw)
    assert text == "the quick brown fox"          # full output used as the answer
    assert cot.NO_MARKER_NOTE in log               # log flags the missing marker
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_cot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2.cot'`.

- [ ] **Step 3: Create `src/htr_sp2/cot.py`**

```python
"""Chain-of-Thought (M2) prompt and output parsing.

M2 reuses the SAME fine-tuned model with a different prompt (prompt-only CoT). The model
was fine-tuned on transcription, not reasoning, so it may not follow the format — the
parser falls back to using the whole output as the answer and notes that in the log, so
CER/WER reflect what the model actually produced.
"""
from __future__ import annotations

# Marker the model is asked to put before its final answer. Tunable (see spec open items).
FINAL_MARKER = "Final:"

# Note added to the log when the model did not emit the marker.
NO_MARKER_NOTE = "[no 'Final:' marker found — using full output as the transcription]"

# Prompt for M2. Asks for a short description of ambiguous strokes, then the final answer
# on its own line prefixed with the marker. Mirrors the SP-1 prompt's plain style.
COT_PROMPT = (
    "transcribe the handwritten text. First briefly describe the distinctive stroke "
    "shapes of any ambiguous characters. Then on a new line write only the final "
    "transcription prefixed exactly with 'Final:'\n"
)


def parse_cot(raw: str) -> tuple[str, str]:
    """Split a raw CoT generation into (final_text, reasoning_log).

    If 'Final:' appears, the text after its LAST occurrence is the answer and everything
    before it is the reasoning log. Otherwise the whole output is the answer and the log
    records that the marker was missing.
    """
    if FINAL_MARKER in raw:
        reasoning, _, final = raw.rpartition(FINAL_MARKER)
        return final.strip(), reasoning.strip()
    stripped = raw.strip()
    return stripped, f"{stripped}\n{NO_MARKER_NOTE}".strip()
```

- [ ] **Step 4: Run tests (cot + config together)**

Run: `pytest tests/test_sp2_cot.py tests/test_sp2_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/cot.py tests/test_sp2_cot.py src/htr_sp2/__init__.py src/htr_sp2/config.py tests/test_sp2_config.py
git commit -m "feat(sp2): CoT prompt + parse_cot, and config module"
```

---

## Task 5: RunPod I/O (shared pure (de)serialization)

**Files:**
- Create: `src/htr_sp2/runpod_io.py`
- Test: `tests/test_sp2_runpod_io.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_runpod_io.py`)

```python
"""runpod_io is the wire format shared by the client (RunPodEngine) and the server
(handler). Round-tripping an image through encode->decode and payload->parse keeps both
sides in sync and is fully testable on CPU."""
import io

from PIL import Image

from htr_sp2 import runpod_io


def _tiny_image():
    return Image.new("RGB", (4, 4), color=(255, 255, 255))


def test_encode_then_decode_roundtrips_size():
    b64 = runpod_io.encode_image(_tiny_image())
    assert isinstance(b64, str) and b64  # non-empty base64 string
    img = runpod_io.decode_image(b64)
    assert img.size == (4, 4)


def test_build_payload_shape():
    payload = runpod_io.build_payload(_tiny_image(), prompt="p", max_new_tokens=7)
    assert set(payload["input"].keys()) == {"image_b64", "prompt", "max_new_tokens"}
    assert payload["input"]["prompt"] == "p"
    assert payload["input"]["max_new_tokens"] == 7


def test_parse_input_returns_image_prompt_tokens():
    payload = runpod_io.build_payload(_tiny_image(), prompt="p", max_new_tokens=7)
    parsed = runpod_io.parse_input(payload)  # server reads {"input": {...}}
    assert parsed["prompt"] == "p"
    assert parsed["max_new_tokens"] == 7
    assert parsed["image"].size == (4, 4)


def test_parse_output_extracts_text():
    assert runpod_io.parse_output({"output": {"text": "hello"}}) == "hello"


def test_parse_output_raises_on_bad_shape():
    import pytest
    with pytest.raises(KeyError):
        runpod_io.parse_output({"unexpected": True})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_runpod_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2.runpod_io'`.

- [ ] **Step 3: Create `src/htr_sp2/runpod_io.py`**

```python
"""Wire format shared by the RunPod client (RunPodEngine) and the server handler.

Keeping (de)serialization here means the two sides can never drift, and every byte of it
is unit-testable on a laptop (no GPU, no network).
"""
from __future__ import annotations

import base64
import io

from PIL import Image


def encode_image(image: Image.Image) -> str:
    """PIL image -> base64-encoded PNG string (safe to embed in JSON)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(image_b64: str) -> Image.Image:
    """Inverse of encode_image: base64 PNG string -> loaded PIL image."""
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw))
    image.load()  # force decode now so errors surface here, not later
    return image


def build_payload(image: Image.Image, prompt: str, max_new_tokens: int) -> dict:
    """Client-side: assemble the RunPod Serverless request body."""
    return {
        "input": {
            "image_b64": encode_image(image),
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
    }


def parse_input(event: dict) -> dict:
    """Server-side: read a RunPod event into the args generation needs."""
    inp = event["input"]
    return {
        "image": decode_image(inp["image_b64"]),
        "prompt": inp["prompt"],
        "max_new_tokens": int(inp.get("max_new_tokens", 64)),
    }


def parse_output(data: dict) -> str:
    """Client-side: pull the transcription text out of a RunPod /runsync response."""
    return data["output"]["text"]
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_sp2_runpod_io.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/runpod_io.py tests/test_sp2_runpod_io.py
git commit -m "feat(sp2): shared runpod_io (image encode/decode + payload (de)serialization)"
```

---

## Task 6: NDJSON event schemas

**Files:**
- Create: `src/htr_sp2/schemas.py`
- Test: `tests/test_sp2_schemas.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_schemas.py`)

```python
"""Each streamed line is one event dict. These builders are the single source of truth for
the NDJSON contract the frontend (SP-4) and batch eval (SP-5) consume."""
from htr_sp2 import schemas


def test_meta_event():
    assert schemas.meta_event("line_01.png", True) == {
        "event": "meta", "filename": "line_01.png", "has_ground_truth": True,
    }


def test_result_event():
    ev = schemas.result_event(
        model="m1", text="the quick brown fox", cer=5.26, wer=25.0,
        latency_seconds=0.78, log="done.", status_tag="Raw Output",
    )
    assert ev == {
        "event": "result", "model": "m1", "text": "the quick brown fox",
        "cer": 5.26, "wer": 25.0, "latency_seconds": 0.78,
        "log": "done.", "status_tag": "Raw Output",
    }


def test_result_event_allows_null_metrics():
    ev = schemas.result_event(
        model="m1", text="x", cer=None, wer=None,
        latency_seconds=0.1, log="done.", status_tag="Raw Output",
    )
    assert ev["cer"] is None and ev["wer"] is None


def test_error_and_done_events():
    assert schemas.error_event("m2", "boom") == {"event": "error", "model": "m2", "message": "boom"}
    assert schemas.done_event() == {"event": "done"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2.schemas'`.

- [ ] **Step 3: Create `src/htr_sp2/schemas.py`**

```python
"""Builders for the NDJSON events streamed by /v1/detect.

Plain dicts (not pydantic) keep it trivial to serialize with json.dumps and to assert on
in tests. These functions are the ONE place the event shape is defined.
"""
from __future__ import annotations


def meta_event(filename: str, has_ground_truth: bool) -> dict:
    """First line: what we are about to process."""
    return {"event": "meta", "filename": filename, "has_ground_truth": has_ground_truth}


def result_event(
    model: str,
    text: str,
    cer: float | None,
    wer: float | None,
    latency_seconds: float,
    log: str,
    status_tag: str,
) -> dict:
    """One scenario's result. cer/wer are None when no ground truth was supplied."""
    return {
        "event": "result",
        "model": model,
        "text": text,
        "cer": cer,
        "wer": wer,
        "latency_seconds": latency_seconds,
        "log": log,
        "status_tag": status_tag,
    }


def error_event(model: str, message: str) -> dict:
    """Emitted when one scenario fails; the stream continues with the next one."""
    return {"event": "error", "model": model, "message": message}


def done_event() -> dict:
    """Final line: the stream is complete."""
    return {"event": "done"}
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_sp2_schemas.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/schemas.py tests/test_sp2_schemas.py
git commit -m "feat(sp2): NDJSON event schema builders"
```

---

## Task 7: Engine interface + FakeEngine + factory

**Files:**
- Create: `src/htr_sp2/engine.py`
- Create: `src/htr_sp2/engines/__init__.py`
- Create: `src/htr_sp2/engines/fake.py`
- Test: `tests/test_sp2_engine.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_engine.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2.engine'`.

- [ ] **Step 3: Create `src/htr_sp2/engine.py`**

```python
"""The inference engine boundary.

Engines are intentionally dumb: given an image, a prompt, and a token cap, return the
model's raw decoded string. All prompt selection and CoT parsing live in the backend, so
swapping engines (fake <-> runpod <-> future local GGUF) never touches that logic.
"""
from __future__ import annotations

from typing import Protocol

from PIL import Image


class EngineError(Exception):
    """Raised when an engine cannot produce a result (network, timeout, bad response)."""


class InferenceEngine(Protocol):
    def run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Return the model's raw decoded transcription for the given prompt."""
        ...


def get_engine(name: str | None = None) -> InferenceEngine:
    """Build the engine named by `name` (default: config.ENGINE)."""
    from htr_sp2 import config

    name = name or config.ENGINE
    if name == "fake":
        from htr_sp2.engines.fake import FakeEngine
        return FakeEngine()
    if name == "runpod":
        from htr_sp2.engines.runpod import RunPodEngine
        return RunPodEngine(
            endpoint_id=config.RUNPOD_ENDPOINT_ID,
            api_key=config.RUNPOD_API_KEY,
            timeout=config.RUNPOD_TIMEOUT_SECONDS,
        )
    raise ValueError(f"unknown engine: {name!r} (expected 'fake' or 'runpod')")
```

- [ ] **Step 4: Create `src/htr_sp2/engines/__init__.py`**

```python
"""Concrete InferenceEngine implementations."""
```

- [ ] **Step 5: Create `src/htr_sp2/engines/fake.py`**

```python
"""A deterministic, GPU-free engine for tests and local development.

Returns scripted outputs in call order (reusing the last once exhausted), records every
call for assertions, and can raise EngineError on chosen call indices to exercise the
orchestrator's per-model error handling.
"""
from __future__ import annotations

from PIL import Image

from htr_sp2.engine import EngineError


class FakeEngine:
    def __init__(self, responses: list[str] | None = None, fail_on: set[int] | None = None):
        # responses consumed in order; the last is reused if more calls come in.
        self.responses = list(responses) if responses else ["the quick brown fox"]
        self.fail_on = set(fail_on) if fail_on else set()
        self.calls: list[dict] = []

    def run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        index = len(self.calls)
        self.calls.append({"image": image, "prompt": prompt, "max_new_tokens": max_new_tokens})
        if index in self.fail_on:
            raise EngineError(f"fake failure on call {index}")
        return self.responses[min(index, len(self.responses) - 1)]
```

- [ ] **Step 6: Run test**

Run: `pytest tests/test_sp2_engine.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/htr_sp2/engine.py src/htr_sp2/engines/__init__.py src/htr_sp2/engines/fake.py tests/test_sp2_engine.py
git commit -m "feat(sp2): engine Protocol, EngineError, FakeEngine, get_engine factory"
```

---

## Task 8: Orchestrator — detect_stream NDJSON generator

**Files:**
- Create: `src/htr_sp2/orchestrator.py`
- Test: `tests/test_sp2_orchestrator.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_orchestrator.py`)

```python
"""detect_stream is the heart of the backend: run M1 then M2 against an engine, compute
CER/WER when ground truth is given, and yield NDJSON lines. We drive it with FakeEngine so
it is deterministic. Latency is timing-based, so we assert its type/sign, not a value."""
import json

from htr_sp2 import config
from htr_sp2.engines.fake import FakeEngine
from htr_sp2.orchestrator import detect_stream


def _events(lines):
    # Each yielded line is a JSON object + newline; parse them back for assertions.
    return [json.loads(line) for line in lines]


def test_stream_emits_meta_two_results_done_in_order():
    eng = FakeEngine(responses=["the quick brown fox", "Reasoning: r\nFinal: the quick brown fox"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert [e["event"] for e in events] == ["meta", "result", "result", "done"]
    assert events[0] == {"event": "meta", "filename": "x.png", "has_ground_truth": False}
    assert events[1]["model"] == "m1" and events[2]["model"] == "m2"


def test_m1_uses_raw_text_and_tag():
    eng = FakeEngine(responses=["the quick brown fox", "Final: ignored"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    m1 = events[1]
    assert m1["text"] == "the quick brown fox"
    assert m1["status_tag"] == config.M1_STATUS_TAG
    assert m1["log"] == "Direct visual token translation completed."


def test_m2_is_parsed_into_text_and_reasoning_log():
    eng = FakeEngine(responses=["x", "Reasoning: loop on d\nFinal: medical"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    m2 = events[2]
    assert m2["text"] == "medical"
    assert m2["log"] == "Reasoning: loop on d"
    assert m2["status_tag"] == config.M2_STATUS_TAG


def test_metrics_null_without_ground_truth():
    eng = FakeEngine(responses=["abc", "Final: abc"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert events[1]["cer"] is None and events[1]["wer"] is None


def test_metrics_computed_with_ground_truth():
    # Perfect match -> 0.0 CER/WER; reuses htr_sp1.metrics (jiwer).
    eng = FakeEngine(responses=["the quick brown fox", "Final: the quick brown fox"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth="the quick brown fox"))
    assert events[1]["cer"] == 0.0 and events[1]["wer"] == 0.0


def test_latency_is_non_negative_float():
    eng = FakeEngine(responses=["a", "Final: a"])
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert isinstance(events[1]["latency_seconds"], float) and events[1]["latency_seconds"] >= 0.0


def test_engine_error_on_one_model_emits_error_and_continues():
    # M1 (call 0) fails; M2 (call 1) still runs. Stream stays alive.
    eng = FakeEngine(responses=["x", "Final: medical"], fail_on={0})
    events = _events(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert [e["event"] for e in events] == ["meta", "error", "result", "done"]
    assert events[1] == {"event": "error", "model": "m1", "message": "fake failure on call 0"}
    assert events[2]["model"] == "m2" and events[2]["text"] == "medical"


def test_engine_called_with_correct_prompts_and_caps():
    eng = FakeEngine(responses=["a", "Final: b"])
    list(detect_stream(eng, image=object(), filename="x.png", ground_truth=None))
    assert eng.calls[0]["prompt"] == config.M1_PROMPT
    assert eng.calls[0]["max_new_tokens"] == config.M1_MAX_NEW_TOKENS
    assert eng.calls[1]["prompt"] == config.M2_PROMPT
    assert eng.calls[1]["max_new_tokens"] == config.M2_MAX_NEW_TOKENS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2.orchestrator'`.

- [ ] **Step 3: Create `src/htr_sp2/orchestrator.py`**

```python
"""The detect flow: run each scenario against the engine and yield NDJSON lines.

M1 is a direct transcription; M2 reuses the same model with the CoT prompt and its output
is parsed into (text, reasoning). CER/WER reuse htr_sp1.metrics. One failing scenario emits
an error event and the stream continues to the next — a dead column never kills the others.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

from htr_sp1.metrics import cer as cer_metric
from htr_sp1.metrics import wer as wer_metric
from htr_sp2 import config, schemas
from htr_sp2.cot import parse_cot
from htr_sp2.engine import EngineError, InferenceEngine

M1_LOG = "Direct visual token translation completed."


@dataclass(frozen=True)
class _ModelSpec:
    """Everything that differs between M1 and M2, in one place."""
    model: str
    prompt: str
    max_new_tokens: int
    status_tag: str


_SPECS = [
    _ModelSpec("m1", config.M1_PROMPT, config.M1_MAX_NEW_TOKENS, config.M1_STATUS_TAG),
    _ModelSpec("m2", config.M2_PROMPT, config.M2_MAX_NEW_TOKENS, config.M2_STATUS_TAG),
]


def _line(event: dict) -> str:
    """Serialize one event dict as a single NDJSON line."""
    return json.dumps(event) + "\n"


def detect_stream(
    engine: InferenceEngine,
    image: Image.Image,
    filename: str,
    ground_truth: str | None,
) -> Iterator[str]:
    """Yield NDJSON lines: meta, one result (or error) per scenario, then done."""
    yield _line(schemas.meta_event(filename, ground_truth is not None))

    for spec in _SPECS:
        try:
            start = time.perf_counter()
            raw = engine.run(image, spec.prompt, spec.max_new_tokens)
            latency = round(time.perf_counter() - start, 3)
        except EngineError as exc:
            yield _line(schemas.error_event(spec.model, str(exc)))
            continue

        if spec.model == "m1":
            text, log = raw.strip(), M1_LOG
        else:
            text, log = parse_cot(raw)

        if ground_truth is not None:
            cer_value = round(cer_metric(ground_truth, text), 2)
            wer_value = round(wer_metric(ground_truth, text), 2)
        else:
            cer_value = wer_value = None

        yield _line(schemas.result_event(
            model=spec.model,
            text=text,
            cer=cer_value,
            wer=wer_value,
            latency_seconds=latency,
            log=log,
            status_tag=spec.status_tag,
        ))

    yield _line(schemas.done_event())
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_sp2_orchestrator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/orchestrator.py tests/test_sp2_orchestrator.py
git commit -m "feat(sp2): detect_stream orchestrator (M1+M2, metrics, NDJSON, per-model errors)"
```

---

## Task 9: FastAPI app — /v1/detect and /health

**Files:**
- Create: `src/htr_sp2/api.py`
- Test: `tests/test_sp2_api.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_api.py`)

```python
"""End-to-end through FastAPI's TestClient using the default fake engine. We post a real
(tiny) PNG so the image-decode path runs, then parse the streamed NDJSON lines."""
import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from htr_sp2.api import app

client = TestClient(app)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_detect_streams_meta_results_done():
    resp = client.post(
        "/v1/detect",
        files={"file": ("line_01.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = _parse_ndjson(resp.text)
    assert [e["event"] for e in events] == ["meta", "result", "result", "done"]
    assert events[0]["filename"] == "line_01.png"
    assert events[0]["has_ground_truth"] is False


def test_detect_with_ground_truth_sets_flag_and_metrics():
    resp = client.post(
        "/v1/detect",
        files={"file": ("line_01.png", _png_bytes(), "image/png")},
        data={"ground_truth": "the quick brown fox"},
    )
    events = _parse_ndjson(resp.text)
    assert events[0]["has_ground_truth"] is True
    # Default FakeEngine returns "the quick brown fox" -> perfect match -> 0.0
    assert events[1]["cer"] == 0.0 and events[1]["wer"] == 0.0


def test_detect_rejects_undecodable_image():
    resp = client.post(
        "/v1/detect",
        files={"file": ("bad.png", b"not really an image", "image/png")},
    )
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2.api'`.

- [ ] **Step 3: Create `src/htr_sp2/api.py`**

```python
"""FastAPI surface for the SP-2 backend.

POST /v1/detect takes a multipart image (+ optional ground_truth) and streams NDJSON
results. The image is validated up front so a bad upload fails with 422 BEFORE the stream
opens; engine failures mid-stream are reported per-model by the orchestrator instead.
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from htr_sp2.engine import get_engine
from htr_sp2.orchestrator import detect_stream

app = FastAPI(title="HTR SP-2 Backend", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/v1/detect")
async def detect(file: UploadFile = File(...), ground_truth: str | None = Form(None)):
    """Run M1 + M2 on the uploaded image and stream NDJSON results."""
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force decode now so a bad image fails here, not mid-stream
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=422, detail="invalid or undecodable image")

    engine = get_engine()
    stream = detect_stream(engine, image, file.filename or "upload", ground_truth)
    return StreamingResponse(stream, media_type="application/x-ndjson")
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_sp2_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/api.py tests/test_sp2_api.py
git commit -m "feat(sp2): FastAPI app with streaming /v1/detect and /health"
```

---

## Task 10: RunPodEngine (client)

**Files:**
- Create: `src/htr_sp2/engines/runpod.py`
- Test: `tests/test_sp2_runpod_engine.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sp2_runpod_engine.py`)

```python
"""RunPodEngine is an HTTP client to a RunPod Serverless /runsync endpoint. We mock HTTP
with pytest-httpx so no GPU/network is needed: assert the request shape and that responses
and failures map correctly."""
import httpx
import pytest
from PIL import Image

from htr_sp2 import engine
from htr_sp2.engines.runpod import RunPodEngine


def _engine():
    return RunPodEngine(endpoint_id="ep123", api_key="key", timeout=5.0)


def _image():
    return Image.new("RGB", (4, 4), (255, 255, 255))


def test_run_posts_payload_and_returns_text(httpx_mock):
    httpx_mock.add_response(json={"output": {"text": "the quick brown fox"}})
    out = _engine().run(_image(), prompt="transcribe\n", max_new_tokens=64)
    assert out == "the quick brown fox"

    request = httpx_mock.get_request()
    assert request.url == "https://api.runpod.ai/v2/ep123/runsync"
    assert request.headers["authorization"] == "Bearer key"
    body = httpx.Request("POST", request.url, content=request.content).read()
    import json
    sent = json.loads(body)
    assert sent["input"]["prompt"] == "transcribe\n"
    assert sent["input"]["max_new_tokens"] == 64
    assert sent["input"]["image_b64"]  # non-empty base64


def test_run_raises_engine_error_on_http_error(httpx_mock):
    httpx_mock.add_response(status_code=500)
    with pytest.raises(engine.EngineError):
        _engine().run(_image(), prompt="p", max_new_tokens=64)


def test_run_raises_engine_error_on_timeout(httpx_mock):
    httpx_mock.add_exception(httpx.TimeoutException("too slow"))
    with pytest.raises(engine.EngineError):
        _engine().run(_image(), prompt="p", max_new_tokens=64)


def test_run_raises_engine_error_on_bad_response_shape(httpx_mock):
    httpx_mock.add_response(json={"unexpected": True})
    with pytest.raises(engine.EngineError):
        _engine().run(_image(), prompt="p", max_new_tokens=64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sp2_runpod_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'htr_sp2.engines.runpod'`.

- [ ] **Step 3: Create `src/htr_sp2/engines/runpod.py`**

```python
"""RunPod Serverless client engine.

The model runs on a RunPod GPU worker (see runpod/handler.py). This class is a thin HTTP
client: it serializes the request via runpod_io, POSTs to /runsync, and extracts the text.
Any HTTP/timeout/shape problem becomes an EngineError so the orchestrator can report it
per-model without crashing the stream.
"""
from __future__ import annotations

import httpx
from PIL import Image

from htr_sp2 import runpod_io
from htr_sp2.engine import EngineError


class RunPodEngine:
    def __init__(self, endpoint_id: str, api_key: str, timeout: float):
        # /runsync blocks until the job finishes and returns the output inline.
        self.url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        payload = runpod_io.build_payload(image, prompt, max_new_tokens)
        try:
            response = httpx.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EngineError(f"RunPod request failed: {exc}") from exc

        try:
            return runpod_io.parse_output(response.json())
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"unexpected RunPod response: {exc}") from exc
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_sp2_runpod_engine.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp2/engines/runpod.py tests/test_sp2_runpod_engine.py
git commit -m "feat(sp2): RunPodEngine HTTP client with EngineError mapping"
```

---

## Task 11: RunPod Serverless handler (server-side, thin)

This runs on the RunPod GPU worker, not locally. The pure (de)serialization is already
tested via `runpod_io`; the generation path needs a real model, so it is integration/manual
(no unit test). Keep it thin.

**Files:**
- Create: `runpod/handler.py`

- [ ] **Step 1: Create `runpod/handler.py`**

```python
"""RunPod Serverless handler — runs on a GPU worker, NOT on the local backend.

Loads the SP-1 fine-tuned model once on cold start, then for each request decodes the
image, runs the supplied prompt through htr_sp1.inference.generate_transcription, and
returns {"text": ...}. The local backend talks to this via RunPodEngine.

Deploy: build an image from requirements-runpod.txt with this as the entrypoint. The
generation path requires a GPU, so it is validated on RunPod (manual/integration), not in
the CPU unit suite. The request/response wire format is unit-tested via htr_sp2.runpod_io.
"""
from __future__ import annotations

import os

import runpod

from htr_sp1.inference import generate_transcription
from htr_sp2 import runpod_io

# Loaded lazily on first request and cached for the worker's lifetime (avoids reloading
# the 3B model on every call). Replace the loader with the SP-1 model assembly used in
# training/export; env vars point at the base model + adapter.
_MODEL = None
_PROCESSOR = None


def _load_model():
    """Load base PaliGemma + LoRA adapter once. Imports are local so the module imports
    cheaply (and so CPU-only tooling never needs torch)."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from peft import PeftModel
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        base_id = os.environ["HTR_BASE_MODEL_ID"]      # e.g. google/paligemma-3b-pt-448
        adapter_id = os.environ["HTR_ADAPTER_ID"]      # HF repo or local path to LoRA adapter
        processor = PaliGemmaProcessor.from_pretrained(base_id)
        base = PaliGemmaForConditionalGeneration.from_pretrained(base_id, device_map="auto")
        _MODEL = PeftModel.from_pretrained(base, adapter_id)
        _PROCESSOR = processor
    return _MODEL, _PROCESSOR


def handler(event: dict) -> dict:
    """RunPod entrypoint: {"input": {image_b64, prompt, max_new_tokens}} -> {"text": ...}."""
    args = runpod_io.parse_input(event)
    model, processor = _load_model()
    text = generate_transcription(
        model,
        processor,
        image=args["image"],
        prompt=args["prompt"],
        max_new_tokens=args["max_new_tokens"],
    )
    return {"text": text}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
```

- [ ] **Step 2: Sanity-check it imports without a GPU**

Run: `python -c "import ast; ast.parse(open('runpod/handler.py').read()); print('handler parses OK')"`
Expected: `handler parses OK` (we only parse — importing would require `runpod`/torch, which are server-side deps).

- [ ] **Step 3: Commit**

```bash
git add runpod/handler.py
git commit -m "feat(sp2): RunPod Serverless handler (server-side model load + generate glue)"
```

---

## Task 12: Docs + full suite green

**Files:**
- Create: `README-sp2.md`

- [ ] **Step 1: Write `README-sp2.md`**

```markdown
# SP-2 — Backend Core

FastAPI backend comparing two HTR scenarios on one handwriting-line image:

- **M1 (Baseline QLoRA):** direct transcription via the SP-1 fine-tuned model.
- **M2 (QLoRA + CoT):** same model, a CoT prompt; reasoning is parsed out of the output.

Results stream back as **NDJSON** (one event per line): `meta`, a `result` (or `error`)
per scenario, then `done`. CER/WER are computed (reusing `htr_sp1.metrics`) only when a
`ground_truth` form field is supplied; otherwise they are `null`.

## Run locally (fake engine — no GPU)

    pip install -r requirements-backend.txt
    HTR_ENGINE=fake uvicorn htr_sp2.api:app --reload --app-dir src

    curl -N -F file=@line_01.png -F ground_truth="the quick brown fox" \
      http://127.0.0.1:8000/v1/detect

## Run against RunPod

Set the engine + credentials, then start the server:

    export HTR_ENGINE=runpod
    export HTR_RUNPOD_ENDPOINT_ID=...    # RunPod Serverless endpoint id
    export HTR_RUNPOD_API_KEY=...
    uvicorn htr_sp2.api:app --app-dir src

## Deploy the RunPod worker

`runpod/handler.py` is the Serverless entrypoint. Build an image from
`requirements-runpod.txt`, set `HTR_BASE_MODEL_ID` and `HTR_ADAPTER_ID`, and point the
endpoint at `handler.handler`. The wire format is in `htr_sp2.runpod_io`.

## Tests

    pytest -q

All backend tests run on CPU (fake engine + mocked HTTP); the GPU generation path is
validated on RunPod.

## Scope

M1 + M2 only. M3/M4 (RAG/pgvector) and the local GGUF engine are separate sub-projects.
```

- [ ] **Step 2: Run the full suite**

Run: `pytest -q`
Expected: all SP-1 and SP-2 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add README-sp2.md
git commit -m "docs(sp2): backend README (run, deploy, scope)"
```

---

## Self-Review Notes

- **Spec coverage:** M1 (Task 8), M2 CoT single-pass + marker parse with fallback (Tasks 4, 8), optional ground_truth → null metrics (Tasks 8, 9), NDJSON streaming (Tasks 6, 8, 9), `POST /v1/detect` multipart + 422 on bad image (Task 9), `GET /health` (Task 9), swappable engine + FakeEngine + RunPodEngine (Tasks 7, 10), RunPod handler both sides (Tasks 5, 10, 11), CER/WER reuse via `htr_sp1.metrics` (Task 8), per-model error continuation (Task 8), SP-1 `prompt` param (Task 2), dependencies (Task 1). All spec §2 decisions and §9 open items are addressed (COT_PROMPT concrete in Task 4; RunPod schema in Tasks 5/10/11; config env names in Task 3).
- **Type consistency:** `engine.run(image, prompt, max_new_tokens) -> str` used identically in FakeEngine, RunPodEngine, and the orchestrator; `parse_cot -> (text, log)`; `runpod_io.build_payload`/`parse_input`/`parse_output` shapes match across client/handler/tests; event dict keys match between `schemas.py` and all consuming tests.
- **No placeholders:** every step has concrete code/commands. The only intentionally-manual path is the GPU generation in `runpod/handler.py`, validated on RunPod (documented), with its wire format unit-tested via `runpod_io`.
