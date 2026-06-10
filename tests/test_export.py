"""Export saves adapter + merged weights to disk, then pushes to the Hub. We test the
path/repo-id logic, the disk-write steps, and the push layer, all with the network mocked.
"""
from unittest.mock import MagicMock

from htr_sp1 import config, export


def test_adapter_and_merged_repo_ids_are_distinct():
    base = "user/paligemma-iam-line-qlora"
    assert export.adapter_repo_id(base) == "user/paligemma-iam-line-qlora-adapter"
    assert export.merged_repo_id(base) == "user/paligemma-iam-line-qlora-merged"


def test_push_folder_creates_repo_then_uploads(monkeypatch):
    import huggingface_hub

    fake_api = MagicMock()
    monkeypatch.setattr(huggingface_hub, "HfApi", MagicMock(return_value=fake_api))

    export.push_folder("/tmp/run/merged", "user/repo-merged", commit_message="run 2026")

    # Repo is created if missing (idempotent), then the folder bytes are uploaded.
    fake_api.create_repo.assert_called_once_with(
        "user/repo-merged", repo_type="model", private=True, exist_ok=True
    )
    fake_api.upload_folder.assert_called_once_with(
        folder_path="/tmp/run/merged", repo_id="user/repo-merged",
        repo_type="model", commit_message="run 2026", allow_patterns=None,
    )


def test_save_adapter_writes_adapter_and_processor():
    model = MagicMock()
    processor = MagicMock()

    result = export.save_adapter(model, processor, "/tmp/run/final_adapter")

    assert result == "/tmp/run/final_adapter"
    model.save_pretrained.assert_called_once_with("/tmp/run/final_adapter")
    processor.save_pretrained.assert_called_once_with("/tmp/run/final_adapter")


def test_adapter_allow_patterns_cover_adapter_and_processor_files():
    # The allowlist must grab the LoRA weights/config AND the processor/tokenizer files, so a
    # checkpoint-sourced push uploads a clean adapter (no optimizer/rng training state).
    pats = export.ADAPTER_ALLOW_PATTERNS
    assert "adapter_model.safetensors" in pats
    assert "adapter_config.json" in pats
    assert any("preprocessor" in p for p in pats)


def test_merge_to_dir_reloads_fullprecision_base_and_saves(monkeypatch):
    import peft
    import transformers

    merged = MagicMock()
    fake_base = MagicMock()
    base_from_pretrained = MagicMock(return_value=fake_base)
    monkeypatch.setattr(
        transformers.PaliGemmaForConditionalGeneration, "from_pretrained", base_from_pretrained
    )

    fake_processor = MagicMock()
    proc_from_pretrained = MagicMock(return_value=fake_processor)
    monkeypatch.setattr(
        transformers.PaliGemmaProcessor, "from_pretrained", proc_from_pretrained
    )

    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = merged
    peft_from_pretrained = MagicMock(return_value=peft_model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", peft_from_pretrained)

    result = export.merge_to_dir("/tmp/run/final_adapter", "/tmp/run/merged")

    assert result == "/tmp/run/merged"
    # Base reloaded at full precision (NOT quantized) from the base model id.
    base_from_pretrained.assert_called_once()
    assert base_from_pretrained.call_args.args[0] == config.BASE_MODEL_ID
    assert "quantization_config" not in base_from_pretrained.call_args.kwargs
    # Adapter attached to the CLEAN base from the given SOURCE dir, then merged.
    assert peft_from_pretrained.call_args.args[0] is fake_base
    assert peft_from_pretrained.call_args.args[1] == "/tmp/run/final_adapter"
    peft_model.merge_and_unload.assert_called_once()
    # Merged model + processor written to the target dir (NOT pushed here).
    merged.save_pretrained.assert_called_once_with("/tmp/run/merged")
    fake_processor.save_pretrained.assert_called_once_with("/tmp/run/merged")
    merged.push_to_hub.assert_not_called()


def _patch_save_and_merge(monkeypatch):
    """Stub the disk-writing steps so save_and_push can be tested with no model/disk."""
    monkeypatch.setattr(export, "save_adapter",
                        lambda m, p, d: d)                       # returns the adapter dir
    monkeypatch.setattr(export, "merge_to_dir",
                        lambda src, d, **k: d)                   # returns the merged dir


def test_save_and_push_happy_path(monkeypatch):
    _patch_save_and_merge(monkeypatch)
    pushed = []
    monkeypatch.setattr(export, "push_folder",
                        lambda d, repo, **k: pushed.append((d, repo)))

    status = export.save_and_push(
        MagicMock(), MagicMock(), output_dir="/run", hub_repo="user/repo",
    )

    assert status["adapter_dir"].endswith("final_adapter")
    assert status["merged_dir"].endswith("merged")
    assert status["pushed"] is True
    assert status["error"] is None
    # adapter dir -> adapter repo (with allow patterns), merged dir -> merged repo.
    assert pushed == [
        ("/run/final_adapter", "user/repo-adapter"),
        ("/run/merged", "user/repo-merged"),
    ]


def test_save_and_push_catches_push_failure(monkeypatch):
    _patch_save_and_merge(monkeypatch)

    def boom(d, repo, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(export, "push_folder", boom)

    status = export.save_and_push(
        MagicMock(), MagicMock(), output_dir="/run", hub_repo="user/repo",
    )

    # Artifacts are still reported (they were written before the push), and the run is NOT lost.
    assert status["adapter_dir"].endswith("final_adapter")
    assert status["merged_dir"].endswith("merged")
    assert status["pushed"] is False
    assert "network down" in status["error"]


def test_save_and_push_skips_push_when_disabled(monkeypatch):
    _patch_save_and_merge(monkeypatch)
    monkeypatch.setattr(export, "push_folder",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not push")))

    status = export.save_and_push(
        MagicMock(), MagicMock(), output_dir="/run", hub_repo="user/repo", push=False,
    )

    assert status["pushed"] is False
    assert status["error"] is None
