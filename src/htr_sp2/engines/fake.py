"""A deterministic, GPU-free engine for tests and local development.

Returns scripted outputs in call order (reusing the last once exhausted), records every
call for assertions, and can raise EngineError on chosen call indices to exercise the
orchestrator's per-model error handling.

Design notes (thesis):
- `FakeEngine` satisfies the `InferenceEngine` Protocol via structural typing — it has
  the right `run` signature and no inheritance is needed.
- `responses` is consumed like a queue; once exhausted the last element is returned
  forever. This lets simple tests pass a single-element list without worrying about
  call counts.
- `fail_on` is a set of zero-based call indices that should raise `EngineError`. Used to
  test that the orchestrator emits the correct error event and continues processing the
  other model rather than crashing the request.
- `calls` is a list of dicts capturing every argument to `run`. Tests use `eng.calls[i]`
  to assert that the orchestrator passes the right image object, prompt string, and token
  cap to each engine call.
"""
from __future__ import annotations

from PIL import Image  # noqa: F401  (kept for type hint parity with InferenceEngine)

from htr_sp2.engine import EngineError


class FakeEngine:
    """Deterministic stand-in for a real inference engine.

    Parameters
    ----------
    responses:
        Ordered list of strings the engine will return, one per call. When the list is
        exhausted, the last string is returned for all subsequent calls.
        Defaults to ``["the quick brown fox"]`` so a bare ``FakeEngine()`` is usable
        without any configuration.
    fail_on:
        Set of zero-based call indices on which ``run`` raises ``EngineError``. Used to
        unit-test the orchestrator's per-model error path without a real network.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        fail_on: set[int] | None = None,
    ) -> None:
        # Convert to a plain list so callers can pass any sequence type.
        self.responses: list[str] = list(responses) if responses else ["the quick brown fox"]

        # Use an empty set as the default; None signals "no failures configured".
        self.fail_on: set[int] = set(fail_on) if fail_on else set()

        # Call log: each entry is {"image": ..., "prompt": ..., "max_new_tokens": ...}.
        # Tests use `eng.calls[i]` to verify what the orchestrator passed to the engine.
        self.calls: list[dict] = []

    def run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Simulate an inference call.

        Records the arguments, raises ``EngineError`` if this call's index is in
        ``fail_on``, otherwise returns the next scripted response.

        Parameters
        ----------
        image:
            Passed through unchanged; stored in ``calls`` so tests can assert identity
            (``eng.calls[0]["image"] is img``).
        prompt:
            The rendered prompt string. Stored in ``calls``.
        max_new_tokens:
            Token cap. Stored in ``calls``.

        Returns
        -------
        str
            The scripted response for this call index, or the last scripted response if
            the list is exhausted.

        Raises
        ------
        EngineError
            If ``len(self.calls)`` (before appending) is in ``self.fail_on``.
        """
        # Capture the call index BEFORE appending so index matches call number (0-based).
        index = len(self.calls)

        # Always record the call — even failing calls are logged so tests can inspect
        # the full call sequence.
        self.calls.append({
            "image": image,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        })

        # Raise before returning if this call index is configured to fail.
        if index in self.fail_on:
            raise EngineError(f"fake failure on call {index}")

        # Return the scripted response for this index; clamp to the last element once
        # the list is exhausted so tests don't need to pre-size the response list.
        return self.responses[min(index, len(self.responses) - 1)]
