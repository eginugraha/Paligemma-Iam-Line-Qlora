"""The inference engine boundary.

Engines are intentionally dumb: given an image, a prompt, and a token cap, return the
model's raw decoded string. All prompt selection and CoT parsing live in the backend, so
swapping engines (fake <-> runpod <-> future local GGUF) never touches that logic.

Design notes (thesis):
- `InferenceEngine` is a `Protocol` (PEP 544 structural subtyping). Implementers do NOT
  inherit from it; they just need a matching `run` signature. This keeps the fake and real
  engines completely decoupled from each other and from the Protocol definition.
- `EngineError` is the single exception type the SP-2 orchestrator catches. Mapping all
  engine-level failures (network error, timeout, bad HTTP status, garbled response) to one
  type simplifies the orchestrator's per-model error-event emission.
- `get_engine` defers all engine imports to the branch they are needed in. The RunPod
  engine module (`engines/runpod.py`) does not exist yet; the lazy import means that
  importing `engine.py` — and running tests against FakeEngine — never triggers that
  missing module.
"""
from __future__ import annotations

from typing import Protocol

from PIL import Image


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class EngineError(Exception):
    """Raised when an engine cannot produce a result (network, timeout, bad response).

    The orchestrator catches this type to emit per-model error events without crashing
    the entire inference request. Any lower-level exception (requests.Timeout, json
    decode errors, HTTP 5xx) should be wrapped in an EngineError before propagating.
    """


# ---------------------------------------------------------------------------
# Protocol (structural interface)
# ---------------------------------------------------------------------------

class InferenceEngine(Protocol):
    """The minimal interface every engine must satisfy.

    A Protocol means any class with a matching `run` method is automatically an
    `InferenceEngine` — no inheritance required. This lets FakeEngine, RunPodEngine,
    and any future local engine (e.g. llama.cpp GGUF) remain fully independent.
    """

    def run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Return the model's raw decoded transcription for the given prompt.

        Parameters
        ----------
        image:
            The handwriting image to transcribe. Already decoded to a PIL Image so the
            engine does not need to know about HTTP multipart or base64 encoding.
        prompt:
            The fully-rendered text prompt (M1 baseline or M2 CoT). Prompt selection
            lives in the backend/orchestrator, not here.
        max_new_tokens:
            Hard cap on generated tokens. M1 uses the SP-1 cap (~64); M2 uses 256 to
            accommodate the reasoning prefix before the 'Final:' answer.

        Returns
        -------
        str
            Raw decoded string exactly as the model outputs it, before any CoT parsing.
            Callers are responsible for stripping / parsing the response.
        """
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_engine(name: str | None = None) -> InferenceEngine:
    """Build and return the engine named by `name` (default: ``config.ENGINE``).

    The function reads ``config.ENGINE`` lazily (inside the call, not at import time) so
    monkeypatching the config attribute in tests works correctly — module-level reads
    would capture the value at import time and ignore later patches.

    Parameters
    ----------
    name:
        Override the engine name. If ``None``, ``config.ENGINE`` is used. Recognised
        values: ``"fake"`` and ``"runpod"``.

    Raises
    ------
    ValueError
        If `name` is not a recognised engine key.
    """
    # Defer config import so patching config.ENGINE in tests works correctly.
    from htr_sp2 import config

    # Resolve the engine name: explicit argument wins; otherwise fall back to config.
    name = name or config.ENGINE

    if name == "fake":
        # FakeEngine: deterministic, no GPU, no network. Used by all unit tests and for
        # local development without cloud credentials.
        from htr_sp2.engines.fake import FakeEngine
        return FakeEngine()

    if name == "runpod":
        # RunPodEngine: calls the RunPod Serverless REST API. Imported lazily so the
        # module being absent (it is a later task) does not break imports of this file.
        from htr_sp2.engines.runpod import RunPodEngine  # type: ignore[import]
        return RunPodEngine(
            endpoint_id=config.RUNPOD_ENDPOINT_ID,
            api_key=config.RUNPOD_API_KEY,
            timeout=config.RUNPOD_TIMEOUT_SECONDS,
        )

    raise ValueError(f"unknown engine: {name!r} (expected 'fake' or 'runpod')")
