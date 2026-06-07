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
    # Merging a PEFT model returns a plain model we then push.
    model.merge_and_unload.return_value = MagicMock()

    export.push_adapter(model, processor, repo_id="user/repo-adapter")
    model.push_to_hub.assert_called_once_with("user/repo-adapter", private=True)
    processor.push_to_hub.assert_called_once_with("user/repo-adapter", private=True)
