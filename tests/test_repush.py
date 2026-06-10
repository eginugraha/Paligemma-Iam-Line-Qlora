"""resolve_repush_plan is PURE: given paths + injected existence/checkpoint lookups, it decides
the adapter source, whether the merged dir must be (re)built, and the target repo ids — with no
model, disk, or network. This is what makes recovery testable on a laptop.
"""
import pytest

from htr_sp1 import repush


def test_uses_final_adapter_when_present():
    plan = repush.resolve_repush_plan(
        "/run", "user/repo",
        exists=lambda p: p in {"/run/final_adapter"},          # merged dir absent
        find_checkpoint=lambda d: None,
    )
    assert plan.adapter_source == "/run/final_adapter"
    assert plan.need_merge is True                              # /run/merged does not exist
    assert plan.merged_dir == "/run/merged"
    assert plan.adapter_repo == "user/repo-adapter"
    assert plan.merged_repo == "user/repo-merged"


def test_skips_merge_when_merged_dir_exists():
    plan = repush.resolve_repush_plan(
        "/run", "user/repo",
        exists=lambda p: p in {"/run/final_adapter", "/run/merged"},
        find_checkpoint=lambda d: None,
    )
    assert plan.need_merge is False


def test_falls_back_to_latest_checkpoint_when_no_final_adapter():
    plan = repush.resolve_repush_plan(
        "/run", "user/repo",
        exists=lambda p: False,                                 # no final_adapter, no merged
        find_checkpoint=lambda d: "/run/checkpoint-300",
    )
    assert plan.adapter_source == "/run/checkpoint-300"
    assert plan.need_merge is True


def test_explicit_adapter_overrides_defaults():
    plan = repush.resolve_repush_plan(
        "/run", "user/repo", adapter="/custom/adapter",
        exists=lambda p: p == "/custom/adapter",
        find_checkpoint=lambda d: "/run/checkpoint-300",
    )
    assert plan.adapter_source == "/custom/adapter"


def test_raises_when_no_adapter_anywhere():
    with pytest.raises(FileNotFoundError) as excinfo:
        repush.resolve_repush_plan(
            "/run", "user/repo",
            exists=lambda p: False,
            find_checkpoint=lambda d: None,
        )
    # The error names both places it looked, so the user knows what to provide.
    assert "final_adapter" in str(excinfo.value)


def test_carries_compute_dtype():
    plan = repush.resolve_repush_plan(
        "/run", "user/repo", compute_dtype="float16",
        exists=lambda p: p == "/run/final_adapter",
        find_checkpoint=lambda d: None,
    )
    assert plan.compute_dtype == "float16"
