# SP-2 — Backend Core: Design

**Date:** 2026-06-08
**Status:** Approved (brainstorming complete) — ready for implementation plan
**Depends on:** SP-1 (model training, merged to `main`). Inference contract: `from htr_sp1.inference import generate_transcription`; metrics: `from htr_sp1.metrics import cer, wer`.
**Blocks:** SP-3 (RAG/pgvector), SP-4 (Svelte frontend), SP-5 (batch eval + dashboard).

---

## 1. Purpose & scope

Build the FastAPI backend that turns one handwriting-line image into a comparative
result for **two scenarios only**:

- **M1 (Baseline QLoRA):** direct visual transcription via the SP-1 fine-tuned model.
- **M2 (QLoRA + CoT):** same fine-tuned model, a *different prompt* that asks the model
  to describe stroke characteristics, then give a final answer. CoT is **prompt-only** —
  no separate model.

Out of scope (explicitly deferred to other sub-projects):

- M3/M4 (RAG / pgvector correction) → SP-3.
- Local GGUF inference engine → later sub-project.
- Svelte frontend → SP-4. Batch evaluation + dashboard → SP-5.

The NDJSON response defined here is the contract the frontend (SP-4) and batch eval (SP-5)
will consume.

## 2. Key decisions (locked during brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Concrete inference engine first | **RunPod / HF transformers**, behind a swappable interface, plus a deterministic `FakeEngine` for tests. Local GGUF deferred. |
| 2 | RunPod scope | SP-2 ships **both** the server-side RunPod Serverless handler **and** the backend client (`RunPodEngine`). |
| 3 | Ground truth & CER/WER | `ground_truth` is an **optional** multipart field. Present → compute CER/WER. Absent → `cer`/`wer` = `null`, text still returned. |
| 4 | Streaming mechanism | **NDJSON** chunked (`StreamingResponse`, one JSON object per line). Chosen over SSE because the request is `POST multipart` (browser `EventSource` is GET-only) and NDJSON is simplest to produce and test. |
| 5 | M2 CoT mechanism | **Single-pass + marker parsing.** Prompt asks for `Reasoning: ... \nFinal: <text>`. Parse text after `Final:` → `text`; the rest → `log`. If the marker is absent (model didn't comply), fallback: whole output → `text`, `log` carries a note. |

Research note: the SP-1 model was fine-tuned *only* on transcription, not reasoning. M2 may
ignore the CoT format or perform worse — for a comparative thesis that is a valid finding,
hence the fallback parsing rather than failing.

## 3. Architecture

New package `htr_sp2`, separate from `htr_sp1`. The engine is intentionally "dumb": it takes
`(image, prompt, max_new_tokens)` and returns the model's **raw decoded string**. All
prompt selection (M1 vs M2) and CoT parsing live in the **backend** (`cot.py`), not in the
RunPod handler. This keeps the handler thin/generic and makes M2 parsing fully testable on CPU.

```
src/htr_sp2/
  __init__.py
  config.py        ENGINE selector, RUNPOD_* settings, timeouts, COT prompt, max tokens
  api.py           FastAPI app; routes: POST /v1/detect, GET /health
  orchestrator.py  detect flow: run M1 then M2, compute metrics, emit NDJSON events
  engine.py        InferenceEngine Protocol + get_engine() factory
  engines/
    __init__.py
    runpod.py      RunPodEngine — HTTP client to RunPod Serverless endpoint
    fake.py        FakeEngine — deterministic, no GPU
  cot.py           COT_PROMPT + parse_cot(raw) -> (text, log)
  schemas.py       NDJSON event models (meta / result / error / done)
runpod/
  handler.py       SERVER-SIDE handler deployed to RunPod; NOT imported by the backend
tests/
  ... (see §7)
```

### 3.1 Engine contract

```python
class InferenceEngine(Protocol):
    def run(self, image: PIL.Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Return the model's raw decoded transcription for the given prompt."""
```

- **RunPodEngine** (`engines/runpod.py`): base64-encodes the image and POSTs
  `{image_b64, prompt, max_new_tokens}` to the RunPod Serverless endpoint, returns the
  response `text`. Configurable timeout (generous — no hard 5s limit). Raises a typed
  engine error on HTTP failure/timeout so the orchestrator can emit a per-model `error`.
- **FakeEngine** (`engines/fake.py`): returns a deterministic string derived from the
  prompt — for M1 a fixed transcription; for M2 a string containing a `Final:` marker so
  the parser path is exercised. No GPU, no network.
- **`get_engine()`** (`engine.py`): factory selecting the implementation from
  `config.ENGINE` (`"runpod"` | `"fake"`).

### 3.2 RunPod handler (`runpod/handler.py`)

Server-side, deployed to RunPod Serverless. Responsibilities:

1. On cold start: load base PaliGemma + LoRA adapter once (module-level), reuse across calls.
2. Per request: decode `image_b64` → PIL, call the SP-1 generation path with the supplied
   `prompt` and `max_new_tokens`, return `{ "text": <raw> }`.

Uses `htr_sp1.inference.generate_transcription` for generation — see §3.3.

### 3.3 Targeted SP-1 improvement

`htr_sp1.inference.generate_transcription` currently hardcodes `config.TRANSCRIPTION_PROMPT`.
Add an optional `prompt: str = config.TRANSCRIPTION_PROMPT` parameter (backward-compatible —
existing callers unaffected) so the handler can drive both the M1 prompt and the M2 CoT
prompt through one function. This is the only change to SP-1.

## 4. Data flow — `POST /v1/detect`

**Request:** `multipart/form-data`
- `file` (required): image (`.png` / `.jpg` / `.jpeg`).
- `ground_truth` (optional): reference string for CER/WER.

**Response:** `text/x-ndjson`, chunked. One JSON object per line, in order:

```
{"event":"meta","filename":"line_01.png","has_ground_truth":true}
{"event":"result","model":"m1","text":"...","cer":5.26,"wer":25.0,"latency_seconds":0.78,"log":"Direct visual token translation completed.","status_tag":"Raw Output"}
{"event":"result","model":"m2","text":"...","cer":15.78,"wer":25.0,"latency_seconds":2.15,"log":"Reasoning: ...","status_tag":"Reasoned"}
{"event":"done"}
```

Orchestrator logic:

```
validate image (decodable PIL) -> on failure: HTTP 422 JSON (before stream opens)
open NDJSON stream:
  emit meta {filename, has_ground_truth}
  for mode in [m1, m2]:
      prompt = TRANSCRIPTION_PROMPT (m1) | COT_PROMPT (m2)
      t0; raw = engine.run(image, prompt, max_new_tokens); latency = now - t0
      if m1: text = raw;  log = "Direct visual token translation completed."
      if m2: text, log = cot.parse_cot(raw)        # marker parse, with fallback
      cer = metrics.cer(gt, text) if gt else None  # likewise wer
      status_tag = "Raw Output" (m1) | "Reasoned" (m2)
      emit result {model, text, cer, wer, latency_seconds, log, status_tag}
  emit done
```

CER/WER reuse `htr_sp1.metrics.cer` / `wer` (jiwer, returns percentage). No new metric code.

`latency_seconds` is measured around the `engine.run` call (includes network for RunPod).

## 5. CoT parsing (`cot.py`)

- `COT_PROMPT`: instructs the model to output `Reasoning: <...>\nFinal: <text>`.
- `parse_cot(raw) -> (text, log)`:
  - If `Final:` marker present → `text` = content after the last `Final:` (stripped);
    `log` = the reasoning portion (everything before that `Final:`, stripped).
  - If absent → `text` = full raw (stripped); `log` = raw + a note that no marker was found
    (so the comparison stays honest and CER/WER reflect what the model actually produced).

## 6. Error handling

| Condition | Behaviour |
|-----------|-----------|
| Missing / undecodable image | `HTTP 422` JSON error, stream never opens. |
| Engine failure for one model (RunPod HTTP error / timeout) | Emit `{"event":"error","model":"m1|m2","message":...}`, **continue** to the next model, then `done`. One failed column does not kill the other. |
| RunPod cold start latency | Generous, configurable client timeout (no hard 5s limit). |

## 7. Testing strategy

All tests run on a CPU laptop, no GPU, no real RunPod calls.

- **Orchestrator** (with `FakeEngine`): NDJSON event order and shape; CER/WER values when GT
  present; `cer`/`wer` = `null` when GT absent; M2 parsing with marker and the no-marker
  fallback; per-model `error` event on engine failure (inject a failing fake) without
  aborting the other model.
- **API** (FastAPI `TestClient`): multipart upload happy path (collect streamed lines);
  `422` on a corrupt/empty image; `GET /health`.
- **RunPodEngine** (mocked HTTP via `pytest-httpx`): asserts request shape
  (`image_b64`, `prompt`, `max_new_tokens`), parses the response `text`, maps
  timeout/HTTP error to the typed engine error.
- **cot.parse_cot** (pure unit): marker present, marker absent (fallback), multiple
  `Final:` occurrences (take the last), whitespace handling.
- **handler.py**: kept thin; the GPU generation path is integration/manual (needs a real
  model). Any pure helper (e.g. request decoding) is unit-tested.

## 8. New dependencies

Backend runtime: `fastapi`, `uvicorn[standard]`, `python-multipart`, `httpx`.
Dev/test: `pytest-httpx`.
Server-side only (RunPod handler, NOT in the local backend env): `runpod` SDK.

(Reuse existing SP-1 pins where shared: `pillow`, `jiwer` via `htr_sp1`.)

## 9. Open items for the implementation plan

- Exact `COT_PROMPT` wording (tune so the fine-tuned model is most likely to emit `Final:`).
- RunPod Serverless request/response JSON schema finalised in `runpod/handler.py` and
  mirrored by `RunPodEngine`.
- Config defaults and env var names (`HTR_ENGINE`, `HTR_RUNPOD_ENDPOINT_ID`,
  `HTR_RUNPOD_API_KEY`, timeout).
