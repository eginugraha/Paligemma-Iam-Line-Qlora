"""Wire format shared by the RunPod client (RunPodEngine) and the server handler.

Keeping (de)serialization here means the two sides can never drift, and every byte of it
is unit-testable on a laptop (no GPU, no network).

Why a dedicated module?
-----------------------
The RunPod Serverless API works by POSTing a JSON body:

    { "input": { "image_b64": "<base64 PNG>", "prompt": "...", "max_new_tokens": 64 } }

The handler on the GPU pod receives the same dict (via ``event["input"]``). If the
client and server each define their own field names, one character typo causes silent
failures. By centralising here we have a single source of truth that both import.

Encoding choice: PNG over JPEG
-------------------------------
We use PNG (lossless) so that the exact pixel values survive the round-trip. JPEG would
introduce compression artefacts before the model even sees the image, which could subtly
degrade transcription quality. The base64 overhead (~33 %) is acceptable given that
handwriting crops are small (typically < 100 KB before encoding).
"""
from __future__ import annotations

import base64
import io

from PIL import Image


# ---------------------------------------------------------------------------
# Image codec
# ---------------------------------------------------------------------------

def encode_image(image: Image.Image) -> str:
    """Encode a PIL image to a base64 PNG string safe to embed in JSON.

    Steps:
      1. Convert to RGB — strips alpha channels that PNG allows but the model
         does not need, and normalises palette-mode images.
      2. Write to an in-memory buffer in PNG format (lossless).
      3. Base64-encode the raw bytes and decode to an ASCII string so it can
         be embedded directly in a JSON value without further escaping.

    Args:
        image: Any PIL/Pillow image object.

    Returns:
        A non-empty ASCII string suitable for use as a JSON string value.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    # b64encode returns bytes; .decode("ascii") gives us a plain Python str
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(image_b64: str) -> Image.Image:
    """Inverse of encode_image: base64 PNG string -> fully loaded PIL image.

    Calling ``.load()`` forces Pillow to decode the compressed PNG data
    immediately rather than lazily. This surfaces any corruption errors here,
    at the boundary, rather than in the middle of inference code.

    Args:
        image_b64: An ASCII base64-encoded PNG string (as produced by
            encode_image).

    Returns:
        A fully decoded PIL Image in whatever mode the PNG stored (RGB after
        encode_image).
    """
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw))
    image.load()  # force decode now so errors surface here, not later
    return image


# ---------------------------------------------------------------------------
# Client-side helpers (build the JSON body before POST-ing to RunPod)
# ---------------------------------------------------------------------------

def build_payload(image: Image.Image, prompt: str, max_new_tokens: int) -> dict:
    """Assemble the RunPod Serverless request body from its logical parts.

    The returned dict is the complete body to POST to the RunPod ``/runsync``
    (or ``/run``) endpoint. RunPod requires everything to live under the
    ``"input"`` key.

    Args:
        image:          The handwriting crop to transcribe.
        prompt:         The text prompt (e.g. a CoT prefix from cot.py).
        max_new_tokens: Maximum tokens the model should generate.

    Returns:
        Dict of the form ``{"input": {"image_b64": ..., "prompt": ...,
        "max_new_tokens": ...}}``.
    """
    return {
        "input": {
            "image_b64": encode_image(image),
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
    }


# ---------------------------------------------------------------------------
# Server-side helpers (parse the RunPod event dict inside the handler)
# ---------------------------------------------------------------------------

def parse_input(event: dict) -> dict:
    """Extract and deserialise the handler's input from a RunPod event dict.

    Inside the GPU handler RunPod passes the full request body as ``event``.
    This function unwraps ``event["input"]`` and converts ``image_b64`` back
    to a PIL image so the handler can call the model directly.

    Args:
        event: The dict RunPod passes to the handler function, shaped as
            ``{"input": {"image_b64": ..., "prompt": ..., "max_new_tokens":
            ...}}``.

    Returns:
        A plain dict with keys:
          - ``"image"``          — PIL Image ready for the model preprocessor
          - ``"prompt"``         — str
          - ``"max_new_tokens"`` — int (defaults to 64 if absent)
    """
    inp = event["input"]
    return {
        "image": decode_image(inp["image_b64"]),
        "prompt": inp["prompt"],
        # int() guards against the value arriving as a JSON number (float in
        # Python) in certain RunPod SDK versions
        "max_new_tokens": int(inp.get("max_new_tokens", 64)),
    }


# ---------------------------------------------------------------------------
# Client-side response parser
# ---------------------------------------------------------------------------

def parse_output(data: dict) -> str:
    """Pull the transcription text out of a RunPod ``/runsync`` response.

    RunPod wraps handler return values under ``"output"``. The handler is
    expected to return ``{"text": "<transcription>"}``.

    Raises:
        KeyError: if ``data`` does not contain ``"output"`` or if
            ``data["output"]`` does not contain ``"text"``. We intentionally
            let KeyError propagate rather than silently returning None so that
            callers can distinguish "the model returned empty text" from
            "the response had an unexpected shape".

    Args:
        data: The parsed JSON response body from RunPod.

    Returns:
        The raw transcription string.
    """
    return data["output"]["text"]
