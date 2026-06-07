"""Shared test doubles so unit tests never download models or hit a GPU.

FakeProcessor/FakeModel mimic just the slices of the transformers API our code touches.
This lets us test our *own* logic (prompt building, input assembly, decoding, aggregation)
deterministically on a laptop.
"""
import pytest


class FakeBatch(dict):
    """Stands in for a transformers BatchEncoding: a dict that also supports `.to(device)`."""

    def to(self, _device):
        return self  # no real tensors to move; just return self so call sites work.


class FakeProcessor:
    """Mimics PaliGemmaProcessor for the calls our code makes."""

    def __init__(self):
        self.last_call = None  # records the most recent kwargs so tests can assert on them.

    def __call__(self, text=None, images=None, suffix=None, return_tensors=None):
        # Record what we were asked to encode; return a minimal fake batch.
        self.last_call = {
            "text": text,
            "images": images,
            "suffix": suffix,
            "return_tensors": return_tensors,
        }
        return FakeBatch(input_ids=[[1, 2, 3]])

    def decode(self, _token_ids, skip_special_tokens=True):
        # Tests inject the desired decoded string via `self.next_decoded`.
        return getattr(self, "next_decoded", "decoded text")


class FakeModel:
    """Mimics a transformers model: `.generate(...)` returns fixed token ids."""

    def __init__(self, generated_ids=None):
        self._generated_ids = generated_ids or [[1, 2, 3, 4]]

    def generate(self, **_kwargs):
        return self._generated_ids


@pytest.fixture
def fake_processor():
    return FakeProcessor()


@pytest.fixture
def fake_model():
    return FakeModel()
