"""Export pushes the adapter + merged weights to the Hub. We test the path/repo-id logic
and that push is invoked correctly, with the network call mocked.
"""
from unittest.mock import MagicMock

from htr_sp1 import export


def test_adapter_and_merged_repo_ids_are_distinct():
    base = "user/paligemma-iam-line-qlora"
    assert export.adapter_repo_id(base) == "user/paligemma-iam-line-qlora-adapter"
    assert export.merged_repo_id(base) == "user/paligemma-iam-line-qlora-merged"


def test_push_model_calls_push_to_hub(monkeypatch):
    model = MagicMock()
    processor = MagicMock()

    export.push_adapter(model, processor, repo_id="user/repo-adapter")
    model.push_to_hub.assert_called_once_with("user/repo-adapter", private=True)
    processor.push_to_hub.assert_called_once_with("user/repo-adapter", private=True)


def test_push_merged_merges_then_pushes():
    model = MagicMock()
    processor = MagicMock()
    merged = MagicMock()
    # merge_and_unload folds the adapter into the base and returns a plain model.
    model.merge_and_unload.return_value = merged

    export.push_merged(model, processor, repo_id="user/repo-merged")

    # The MERGED model must be pushed, NOT the original PEFT-wrapped `model`.
    model.merge_and_unload.assert_called_once()
    merged.push_to_hub.assert_called_once_with("user/repo-merged", private=True)
    model.push_to_hub.assert_not_called()
    processor.push_to_hub.assert_called_once_with("user/repo-merged", private=True)
