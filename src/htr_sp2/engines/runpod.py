"""RunPod Serverless client engine.

The model runs on a RunPod GPU worker (see rp_handler.py at the repo root). This class is a
thin HTTP client: it serializes the request via runpod_io, POSTs to /runsync, extracts text.
Any HTTP/timeout/shape problem becomes an EngineError so the orchestrator can report it
per-model without crashing the stream.

Architecture context (thesis):
-------------------------------
SP-2 compares four inference scenarios. For the PaliGemma-based scenarios (M1 and M2),
the model runs on a GPU pod via RunPod Serverless. This engine is the client-side half of
that bridge; the server-side handler lives in rp_handler.py at the repository root.

The /runsync endpoint (as opposed to /run + /status polling) means the HTTP POST blocks
until the job is done and the output is embedded in the response body. This keeps the
client logic simple: one POST, one JSON response.

Failure modes and mapping to EngineError:
------------------------------------------
1. HTTP errors (4xx/5xx): httpx.raise_for_status() raises httpx.HTTPStatusError, a
   subclass of httpx.HTTPError — caught by the except httpx.HTTPError clause.
2. Network timeout: httpx raises httpx.TimeoutException, also a subclass of
   httpx.HTTPError — same except clause covers it.
3. Malformed/unexpected response body: parse_output() raises KeyError or TypeError if
   "output" or "text" are missing — caught by the second except clause.

All three are wrapped as EngineError so the orchestrator's single try/except block can
emit a per-model error event without differentiating transport vs. shape failures.
"""
from __future__ import annotations

import httpx
from PIL import Image

from htr_sp2 import runpod_io
from htr_sp2.engine import EngineError


class RunPodEngine:
    """HTTP client for a single RunPod Serverless inference endpoint.

    This class satisfies the InferenceEngine Protocol (structural subtyping, PEP 544):
    it implements run(image, prompt, max_new_tokens) -> str without inheriting from
    InferenceEngine. This keeps it decoupled from the Protocol and from FakeEngine.

    Attributes:
        url:     The full /runsync URL for this endpoint (built once in __init__).
        headers: HTTP headers dict containing the Bearer token, reused for every request.
        timeout: Seconds to wait before raising a TimeoutException (passed to httpx).
    """

    def __init__(self, endpoint_id: str, api_key: str, timeout: float):
        """Construct the engine from RunPod credentials.

        Args:
            endpoint_id: The RunPod Serverless endpoint ID (e.g. "abc123xyz").
                         Available in the RunPod dashboard under Serverless > Endpoints.
            api_key:     The RunPod API key for authentication. Sent as a Bearer token.
            timeout:     Total seconds to wait for a /runsync response before giving up.
                         Should be >= the model's typical inference time + network RTT.
        """
        # Build the full URL once so every run() call doesn't do string formatting.
        # /runsync blocks on the server until the job finishes, returning output inline.
        self.url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

        # HTTP Authorization header. Note: RunPod expects "Bearer <key>" (case-sensitive
        # in the header value, but HTTP header *names* are case-insensitive).
        self.headers = {"Authorization": f"Bearer {api_key}"}

        # Timeout in seconds for the blocking /runsync call. httpx accepts a float and
        # applies it as a combined connect+read timeout.
        self.timeout = timeout

    def run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Send an inference request to RunPod and return the transcription text.

        The method:
          1. Serialises (image, prompt, max_new_tokens) into the RunPod JSON payload
             using runpod_io.build_payload — the shared wire-format module.
          2. POSTs to /runsync using httpx (synchronous). This blocks until the GPU
             worker finishes or the timeout fires.
          3. Calls raise_for_status() to surface HTTP-level errors (4xx, 5xx).
          4. Parses the response JSON with runpod_io.parse_output to extract the text.

        Args:
            image:          Handwriting crop as a PIL Image. Already decoded from the
                            multipart/base64 upload earlier in the request pipeline.
            prompt:         Fully-rendered prompt string (M1 baseline or M2 CoT prefix).
            max_new_tokens: Hard cap on generated tokens — 64 for M1, 256 for M2.

        Returns:
            The raw decoded transcription string, exactly as the model produced it,
            before any CoT parsing or post-processing.

        Raises:
            EngineError: On any of the following:
                - HTTP error status (4xx / 5xx) — raise_for_status fires httpx.HTTPStatusError
                - Network timeout — httpx raises httpx.TimeoutException
                - Malformed response body — parse_output raises KeyError/TypeError
        """
        # Build the JSON payload. runpod_io.build_payload encodes the image as base64
        # PNG and wraps everything under the "input" key that RunPod requires.
        payload = runpod_io.build_payload(image, prompt, max_new_tokens)

        try:
            # httpx.post is a convenience function for a single synchronous POST.
            # We don't reuse a Client here because each run() is independent and the
            # engine may be called from multiple threads in future; a module-level
            # Client would require careful lifecycle management.
            response = httpx.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            # raise_for_status() raises httpx.HTTPStatusError (subclass of HTTPError)
            # for any 4xx or 5xx response. 2xx and 3xx pass through silently.
            response.raise_for_status()

        except httpx.HTTPError as exc:
            # Catches both httpx.HTTPStatusError (bad status) and httpx.TimeoutException
            # (network timeout) because both are subclasses of httpx.HTTPError.
            # We wrap the original exception as the cause so the full traceback is
            # available in logs, while presenting a clean EngineError to the orchestrator.
            raise EngineError(f"RunPod request failed: {exc}") from exc

        # Parse the response body. parse_output dereferences data["output"]["text"],
        # raising KeyError if either key is absent. We catch KeyError, TypeError
        # (e.g. if data["output"] is not a dict), and ValueError to normalise all
        # unexpected-shape scenarios into a single EngineError.
        try:
            return runpod_io.parse_output(response.json())
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"unexpected RunPod response: {exc}") from exc
