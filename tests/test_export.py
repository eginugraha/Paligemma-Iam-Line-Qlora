"""Export pushes the adapter + merged weights to the Hub. We test the path/repo-id logic
and that push is invoked correctly, with the network call mocked.
"""
from unittest.mock import MagicMock

from htr_sp1 import config, export


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


def test_push_merged_reloads_fullprecision_base_then_merges(monkeypatch):
    # push_merged must NOT merge the in-memory 4-bit model (that rounds the adapter into 4-bit
    # weights and corrupts generations). It should dump the adapter, reload the base at full
    # precision, attach the adapter there, merge, and push the result.
    import peft
    import transformers

    model = MagicMock()       # the in-memory 4-bit PEFT model: ONLY the adapter source
    processor = MagicMock()
    merged = MagicMock()

    # Fake the freshly reloaded full-precision base (network/model load mocked out).
    fake_base = MagicMock()
    base_from_pretrained = MagicMock(return_value=fake_base)
    monkeypatch.setattr(
        transformers.PaliGemmaForConditionalGeneration, "from_pretrained", base_from_pretrained
    )

    # Fake attaching the adapter to that base; merge_and_unload returns the merged model.
    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = merged
    peft_from_pretrained = MagicMock(return_value=peft_model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", peft_from_pretrained)

    export.push_merged(model, processor, repo_id="user/repo-merged")

    # The in-memory 4-bit model must NOT be merged.
    model.merge_and_unload.assert_not_called()
    # Adapter dumped to a scratch dir; base reloaded at full precision (NOT quantized).
    model.save_pretrained.assert_called_once()
    base_from_pretrained.assert_called_once()
    assert base_from_pretrained.call_args.args[0] == config.BASE_MODEL_ID
    assert "quantization_config" not in base_from_pretrained.call_args.kwargs
    # Adapter attached to the CLEAN base, then merged.
    peft_from_pretrained.assert_called_once()
    assert peft_from_pretrained.call_args.args[0] is fake_base
    peft_model.merge_and_unload.assert_called_once()
    # The MERGED model is pushed, NOT the original PEFT-wrapped `model`.
    merged.push_to_hub.assert_called_once_with("user/repo-merged", private=True)
    model.push_to_hub.assert_not_called()
    processor.push_to_hub.assert_called_once_with("user/repo-merged", private=True)
