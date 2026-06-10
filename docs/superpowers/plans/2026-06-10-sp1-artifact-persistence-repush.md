# SP-1 Artifact Persistence & Re-push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SP-1 training artifacts (LoRA adapter + merged fp16 model) durable on local disk *before* any Hub push, make a push failure non-fatal/recoverable, and add a standalone re-push script that recovers a "trained but not pushed" run with no retraining.

**Architecture:** Refactor `htr_sp1.export` to separate **save** / **merge** / **push** into single-purpose functions, where push uploads a directory via `huggingface_hub.HfApi.upload_folder` (no model loaded into RAM). Training (`htr_sp1.cli`) writes `final_adapter/` and `merged/` to the run's `output_dir`, then pushes inside a try/except. A new pure helper `htr_sp1.repush.resolve_repush_plan` + thin CLI `scripts/repush_sp1.py` re-push from disk, falling back to the latest Trainer checkpoint.

**Tech Stack:** Python, pytest, `unittest.mock` (mock the Hub + heavy transformers/peft loads — no GPU/network in CI), `huggingface_hub`, `transformers`, `peft`.

**Reference spec:** `docs/superpowers/specs/2026-06-10-sp1-artifact-persistence-repush-design.md`

**Conventions (match existing SP-1):** heavy module/function docstrings + inline comments (thesis must be explainable); heavy imports (`torch`/`transformers`/`peft`/`huggingface_hub`) are done lazily *inside* functions so the package imports instantly and unit tests stay on CPU; tests live in `tests/` and run with plain `pytest`. Existing export/merge logic and its safety rationale (merge on a CPU full-precision base, never the 4-bit model) must be preserved.

---

## File Structure

```
src/htr_sp1/
  export.py    # REWRITE: ADAPTER_ALLOW_PATTERNS + save_adapter + merge_to_dir + push_folder
               #          + save_and_push; REMOVE push_adapter/push_merged
  repush.py    # NEW: RepushPlan dataclass + pure resolve_repush_plan(...)
  cli.py       # MODIFY step 6 of main(): save-to-disk then push via save_and_push (failure-safe)
scripts/
  repush_sp1.py  # NEW: thin recovery CLI over htr_sp1.repush + export
tests/
  test_export.py   # REWRITE: drop old push tests; add save_adapter/merge_to_dir/push_folder/save_and_push
  test_repush.py   # NEW: pure resolve_repush_plan tests (temp dir, injected predicates)
```

---

## Task 1: `push_folder` — the single Hub upload point

**Files:**
- Modify: `src/htr_sp1/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_export.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py::test_push_folder_creates_repo_then_uploads -v`
Expected: FAIL with `AttributeError: module 'htr_sp1.export' has no attribute 'push_folder'`

- [ ] **Step 3: Write minimal implementation** (add to `src/htr_sp1/export.py`)

```python
def push_folder(local_dir, repo_id, *, commit_message=None, private=True, allow_patterns=None):
    """Upload a local directory to a Hub repo (the ONE place a push happens).

    Uploading to an existing repo adds a new commit (same-named files are overwritten, but the
    Hub keeps history, so it is reversible). No model is loaded into memory — we ship the bytes
    already written to disk, which is fast and cheap even for the ~5.8 GB merged model.

    Args:
        local_dir: Directory whose contents are uploaded (e.g. an adapter or merged-model dir).
        repo_id: Target repo id (use adapter_repo_id(...) / merged_repo_id(...)).
        commit_message: Optional message recorded on the Hub commit (handy for run provenance).
        private: Create the repo private (thesis artifact, not public) if it does not exist.
        allow_patterns: Optional list of globs to restrict which files are uploaded (used to push
                        ONLY adapter/processor files when the source is a full Trainer checkpoint).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    # exist_ok=True: re-pushing to an existing repo is normal (recovery), not an error.
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir, repo_id=repo_id, repo_type="model",
        commit_message=commit_message, allow_patterns=allow_patterns,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export.py::test_push_folder_creates_repo_then_uploads -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/export.py tests/test_export.py
git commit -m "feat(sp1): push_folder — single Hub upload point via upload_folder"
```

---

## Task 2: `ADAPTER_ALLOW_PATTERNS` + `save_adapter`

**Files:**
- Modify: `src/htr_sp1/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_export.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py::test_save_adapter_writes_adapter_and_processor -v`
Expected: FAIL with `AttributeError: module 'htr_sp1.export' has no attribute 'save_adapter'`

- [ ] **Step 3: Write minimal implementation** (add to `src/htr_sp1/export.py`)

```python
# Files that make up a self-contained, loadable adapter (LoRA weights + its config + the
# processor/tokenizer). When the push source is a FULL Trainer checkpoint (recovery fallback),
# this allowlist uploads ONLY these to the adapter repo and skips optimizer/scheduler/rng state.
ADAPTER_ALLOW_PATTERNS = [
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def save_adapter(model, processor, adapter_dir):
    """Write the trained LoRA adapter (+ processor) to *adapter_dir* and return the path.

    PEFT's save_pretrained writes ONLY the small adapter weights, not the frozen base. The
    processor is saved alongside so the directory is self-contained for loading/pushing.

    Args:
        model: The PEFT-wrapped model whose adapter weights are written.
        processor: The matching processor.
        adapter_dir: Destination directory (e.g. <output_dir>/final_adapter).
    """
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    return adapter_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export.py -k "save_adapter or adapter_allow_patterns" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/export.py tests/test_export.py
git commit -m "feat(sp1): save_adapter + ADAPTER_ALLOW_PATTERNS"
```

---

## Task 3: `merge_to_dir` — merge from an adapter source to disk

**Files:**
- Modify: `src/htr_sp1/export.py`
- Test: `tests/test_export.py`

This reuses the existing clean-merge rationale (load the base at full precision on CPU, attach
the adapter there, merge — NEVER merge the 4-bit model) but takes the adapter from a **path/hub
id** and **writes the result to disk** instead of pushing.

- [ ] **Step 1: Write the failing test** (append to `tests/test_export.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py::test_merge_to_dir_reloads_fullprecision_base_and_saves -v`
Expected: FAIL with `AttributeError: module 'htr_sp1.export' has no attribute 'merge_to_dir'`

- [ ] **Step 3: Write minimal implementation** (add to `src/htr_sp1/export.py`)

```python
def merge_to_dir(adapter_source, merged_dir, *, compute_dtype="bfloat16"):
    """Merge a trained adapter into a full-precision base and write the self-contained model.

    CRITICAL (unchanged from the original push_merged rationale): we do NOT merge the in-memory
    4-bit model — folding the adapter into 4-bit weights rounds and visibly corrupts generations.
    Instead we load the base at full precision on CPU (no quantization, no device_map, so we do
    not fight any resident training model for GPU memory), attach the adapter from *adapter_source*,
    merge there, and save to disk. The processor is loaded from the base id (it is never
    fine-tuned) so this works whether *adapter_source* is a final_adapter dir or a raw checkpoint.

    Args:
        adapter_source: Path (or Hub id) of the trained LoRA adapter.
        merged_dir: Destination directory for the merged fp16 model + processor.
        compute_dtype: Full-precision dtype to load the base in ("bfloat16" on Ampere/Ada).

    Returns:
        merged_dir.
    """
    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from peft import PeftModel

    from . import config

    base = PaliGemmaForConditionalGeneration.from_pretrained(
        config.BASE_MODEL_ID, torch_dtype=getattr(torch, compute_dtype),
    )
    merged = PeftModel.from_pretrained(base, adapter_source).merge_and_unload()
    merged.save_pretrained(merged_dir)

    # Processor comes from the base model id (identical, never trained) so a checkpoint source
    # without a saved processor still produces a self-contained merged dir.
    processor = PaliGemmaProcessor.from_pretrained(config.BASE_MODEL_ID)
    processor.save_pretrained(merged_dir)
    return merged_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export.py::test_merge_to_dir_reloads_fullprecision_base_and_saves -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/export.py tests/test_export.py
git commit -m "feat(sp1): merge_to_dir — clean full-precision merge written to disk"
```

---

## Task 4: `save_and_push` — durable-first orchestration

**Files:**
- Modify: `src/htr_sp1/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_export.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py -k save_and_push -v`
Expected: FAIL with `AttributeError: module 'htr_sp1.export' has no attribute 'save_and_push'`

- [ ] **Step 3: Write minimal implementation** (add to `src/htr_sp1/export.py`)

```python
def save_and_push(model, processor, *, output_dir, hub_repo, compute_dtype="bfloat16", push=True):
    """Persist artifacts to disk, then (optionally) push them — the SP-1 definition of done.

    Order matters: the adapter and merged model are written to disk BEFORE any network call, so a
    push failure never loses the run — it can be re-pushed later from disk (scripts/repush_sp1.py).
    Only the push step may fail; it is caught and reported via the returned status.

    Args:
        model: The PEFT-wrapped model (adapter source).
        processor: The matching processor.
        output_dir: Run directory; artifacts go to <output_dir>/final_adapter and <output_dir>/merged.
        hub_repo: Base Hub repo id; adapter/merged are pushed to its -adapter / -merged repos.
        compute_dtype: Full-precision dtype for the merge ("bfloat16" on Ampere/Ada, "float16" on T4).
        push: When False (e.g. --no-push), write both dirs but skip the network entirely.

    Returns:
        {"adapter_dir": str, "merged_dir": str, "pushed": bool, "error": str | None}.
    """
    import os

    adapter_dir = save_adapter(model, processor, os.path.join(output_dir, "final_adapter"))
    merged_dir = merge_to_dir(adapter_dir, os.path.join(output_dir, "merged"),
                              compute_dtype=compute_dtype)

    status = {"adapter_dir": adapter_dir, "merged_dir": merged_dir, "pushed": False, "error": None}
    if not push:
        return status

    try:
        push_folder(adapter_dir, adapter_repo_id(hub_repo), allow_patterns=ADAPTER_ALLOW_PATTERNS)
        push_folder(merged_dir, merged_repo_id(hub_repo))
        status["pushed"] = True
    except Exception as exc:  # network/auth failure must not lose the on-disk run
        status["error"] = str(exc)
    return status
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export.py -k save_and_push -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/export.py tests/test_export.py
git commit -m "feat(sp1): save_and_push — durable-first artifacts, failure-tolerant push"
```

---

## Task 5: Remove old push functions; wire `cli.py` to `save_and_push`

**Files:**
- Modify: `src/htr_sp1/export.py` (remove `push_adapter`, `push_merged`)
- Modify: `tests/test_export.py` (remove the two old push tests)
- Modify: `src/htr_sp1/cli.py` (step 6 of `main`)

- [ ] **Step 1: Remove the obsolete functions and their tests**

In `src/htr_sp1/export.py`, delete the `push_adapter(...)` and `push_merged(...)` function
definitions (their behaviour is now covered by `save_adapter` + `merge_to_dir` + `push_folder`).
Keep `adapter_repo_id` / `merged_repo_id`.

In `tests/test_export.py`, delete `test_push_model_calls_push_to_hub` and
`test_push_merged_reloads_fullprecision_base_then_merges` (they reference the removed functions).
Keep `test_adapter_and_merged_repo_ids_are_distinct` and all Task 1–4 tests.

- [ ] **Step 2: Run export tests to verify nothing references the removed functions**

Run: `python -m pytest tests/test_export.py -v`
Expected: PASS (only the repo-id test + the new save/merge/push/save_and_push tests run)

- [ ] **Step 3: Rewire `cli.py` step 6**

In `src/htr_sp1/cli.py`, replace the entire `# 6. Export adapter + merged ...` block (the
`if not rc.no_push:` block that calls `export.push_adapter` / `export.push_merged` and then does
reload-validation) with:

```python
    # 6. Persist artifacts to disk (definition of done), THEN push. Artifacts are written before
    #    any network call, so a push failure never loses the run — re-push later from disk.
    status = export.save_and_push(
        model, processor,
        output_dir=rc.output_dir, hub_repo=rc.hub_repo,
        compute_dtype="bfloat16" if rc.bf16 else "float16",
        push=not rc.no_push,
    )
    print(f"[SP-1] saved adapter -> {status['adapter_dir']}")
    print(f"[SP-1] saved merged  -> {status['merged_dir']}")

    if rc.no_push:
        print("[SP-1] --no-push: wrote artifacts to disk, skipped Hub upload and reload-validation.")
    elif not status["pushed"]:
        # Push failed but the run is safe on disk. Print the exact recovery command.
        print(f"[SP-1] PUSH FAILED: {status['error']}")
        print("[SP-1] artifacts are safe on disk. Re-push later with:")
        print(f"    python scripts/repush_sp1.py --output-dir {rc.output_dir} --hub-repo {rc.hub_repo}")
    else:
        print("[SP-1] pushed:", export.adapter_repo_id(rc.hub_repo),
              "and", export.merged_repo_id(rc.hub_repo))

        # Reload the MERGED model fresh from the Hub and transcribe a few test images.
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        rid = export.merged_repo_id(rc.hub_repo)
        v_model = PaliGemmaForConditionalGeneration.from_pretrained(rid, device_map="auto")
        v_proc = PaliGemmaProcessor.from_pretrained(rid)
        print("[SP-1] reload-validation:")
        for i in range(3):
            img = ds["test"][i]["image"]
            print("  GT :", ds["test"][i]["text"])
            print("  OUT:", inference.generate_transcription(v_model, v_proc, img))
```

- [ ] **Step 4: Run the CLI tests + full suite (no regressions)**

Run: `python -m pytest tests/test_cli.py tests/test_export.py -v`
Expected: PASS (pure CLI helper tests unchanged; export tests green)

Run: `python -m pytest -q`
Expected: PASS (all SP-1/SP-2/SP-3 tests green; pgvector DB test skipped)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/export.py tests/test_export.py src/htr_sp1/cli.py
git commit -m "refactor(sp1): cli uses save_and_push; drop push_adapter/push_merged"
```

---

## Task 6: `htr_sp1.repush` — pure re-push planner

**Files:**
- Create: `src/htr_sp1/repush.py`
- Test: `tests/test_repush.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repush.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_repush.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'htr_sp1.repush'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/htr_sp1/repush.py
"""Plan a re-push (recovery) without touching the model, disk writes, or network.

`resolve_repush_plan` is pure logic: it decides WHICH adapter to push (an explicit override, else
the run's final_adapter, else the latest Trainer checkpoint), whether the merged model still needs
to be built, and the target repo ids. The thin CLI (scripts/repush_sp1.py) executes the plan with
the real export functions. Existence and checkpoint lookups are injected so this is unit-testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class RepushPlan:
    """The fully-resolved inputs for a re-push."""

    adapter_source: str   # dir (or hub id) of the adapter to push / merge from
    merged_dir: str       # where the merged model lives / will be written
    need_merge: bool      # True if merged_dir must be built before pushing
    adapter_repo: str     # target Hub repo for the adapter
    merged_repo: str      # target Hub repo for the merged model
    compute_dtype: str    # full-precision dtype for the merge


def resolve_repush_plan(
    output_dir: str,
    hub_repo: str,
    *,
    adapter: Optional[str] = None,
    compute_dtype: str = "bfloat16",
    exists: Callable[[str], bool] = os.path.exists,
    find_checkpoint: Callable[[str], Optional[str]] = None,
) -> RepushPlan:
    """Resolve where to read the adapter from and what still needs building/pushing.

    Adapter source precedence: explicit *adapter* > <output_dir>/final_adapter > latest Trainer
    checkpoint (via *find_checkpoint*). Raises FileNotFoundError if none exist.

    Args:
        output_dir: The run directory holding final_adapter/, merged/, checkpoint-*/.
        hub_repo: Base Hub repo id (adapter/merged repos are derived from it).
        adapter: Optional explicit adapter dir/hub-id override.
        compute_dtype: Full-precision dtype for the merge step.
        exists: Predicate for path existence (injected for tests).
        find_checkpoint: Returns the latest Trainer checkpoint dir or None (injected for tests).
    """
    from .export import adapter_repo_id, merged_repo_id

    final_adapter = os.path.join(output_dir, "final_adapter")
    merged_dir = os.path.join(output_dir, "merged")

    if adapter is not None:
        if not exists(adapter):
            raise FileNotFoundError(f"--adapter path does not exist: {adapter}")
        adapter_source = adapter
    elif exists(final_adapter):
        adapter_source = final_adapter
    else:
        checkpoint = find_checkpoint(output_dir) if find_checkpoint is not None else None
        if checkpoint is None:
            raise FileNotFoundError(
                f"No adapter found: neither {final_adapter} nor any Trainer checkpoint in "
                f"{output_dir}. Pass --adapter explicitly or re-run training."
            )
        adapter_source = checkpoint

    return RepushPlan(
        adapter_source=adapter_source,
        merged_dir=merged_dir,
        need_merge=not exists(merged_dir),
        adapter_repo=adapter_repo_id(hub_repo),
        merged_repo=merged_repo_id(hub_repo),
        compute_dtype=compute_dtype,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_repush.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/htr_sp1/repush.py tests/test_repush.py
git commit -m "feat(sp1): resolve_repush_plan — pure re-push planner (adapter/merge/repo resolution)"
```

---

## Task 7: `scripts/repush_sp1.py` — recovery CLI

**Files:**
- Create: `scripts/repush_sp1.py`

This is a thin wrapper (not unit-tested — same convention as `scripts/eval_sp1.py` and
`scripts/ingest_sp3.py`); all branching logic is in the already-tested `resolve_repush_plan`.

- [ ] **Step 1: Write the script**

```python
# scripts/repush_sp1.py
#!/usr/bin/env python
"""SP-1 recovery CLI: re-push a trained run's adapter + merged model to the Hub from disk.

Use this when training finished but the push failed (or to re-publish an existing run). It reads
the adapter from <output_dir>/final_adapter (falling back to the latest Trainer checkpoint),
builds the merged model only if <output_dir>/merged is missing, and pushes both to the Hub. No
retraining. HTR_PG_DSN/HTR_HUB_REPO_ID/HF_TOKEN are read from the shell or a local .env.

Usage:
    export HTR_HUB_REPO_ID="your-hf-username/paligemma-iam-line-qlora"
    huggingface-cli login                 # or set HF_TOKEN in .env
    python scripts/repush_sp1.py --output-dir outputs/sp1
"""
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htr_sp1 import config, export, repush  # noqa: E402
from htr_sp1.train import find_resume_checkpoint  # noqa: E402


def main() -> None:
    import os

    p = argparse.ArgumentParser(description="Re-push a trained SP-1 run (adapter + merged) to the Hub.")
    p.add_argument("--output-dir", default=os.environ.get("HTR_OUTPUT_DIR", config.OUTPUT_DIR),
                   help="Run dir holding final_adapter/ and merged/ (default: $HTR_OUTPUT_DIR/config).")
    p.add_argument("--hub-repo", default=os.environ.get("HTR_HUB_REPO_ID", config.HF_HUB_REPO_ID),
                   help="Base Hub repo id (default: $HTR_HUB_REPO_ID/config).")
    p.add_argument("--adapter", default=None,
                   help="Explicit adapter dir/hub-id (default: <output_dir>/final_adapter or checkpoint).")
    p.add_argument("--compute-dtype", default="bfloat16",
                   help="Full-precision dtype for the merge step (bfloat16 on Ampere/Ada).")
    args = p.parse_args()

    plan = repush.resolve_repush_plan(
        args.output_dir, args.hub_repo,
        adapter=args.adapter, compute_dtype=args.compute_dtype,
        find_checkpoint=find_resume_checkpoint,
    )
    print(f"[SP-1 repush] adapter_source={plan.adapter_source}  need_merge={plan.need_merge}")

    if plan.need_merge:
        print(f"[SP-1 repush] building merged model -> {plan.merged_dir} ...")
        export.merge_to_dir(plan.adapter_source, plan.merged_dir, compute_dtype=plan.compute_dtype)
    else:
        print(f"[SP-1 repush] reusing existing merged dir {plan.merged_dir}")

    print(f"[SP-1 repush] pushing adapter -> {plan.adapter_repo}")
    export.push_folder(plan.adapter_source, plan.adapter_repo,
                       allow_patterns=export.ADAPTER_ALLOW_PATTERNS)
    print(f"[SP-1 repush] pushing merged  -> {plan.merged_repo}")
    export.push_folder(plan.merged_dir, plan.merged_repo)
    print("[SP-1 repush] done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports/compiles (no GPU/network needed)**

Run: `python -m py_compile scripts/repush_sp1.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Run the full suite (no regressions)**

Run: `python -m pytest -q`
Expected: PASS (all SP-1/SP-2/SP-3 tests green; pgvector DB test skipped)

- [ ] **Step 4: Commit**

```bash
git add scripts/repush_sp1.py
git commit -m "feat(sp1): repush_sp1 recovery CLI (re-push adapter + merged from disk)"
```

---

## Post-implementation (manual, on the training server)

These need the trained run + a live Hub, so they run outside the TDD loop:

1. After the next training run, confirm `<output_dir>/final_adapter/` and `<output_dir>/merged/`
   exist on disk and the push succeeded (or the recovery line was printed).
2. To recover the previous "trained but not pushed" run: on the server that still has its
   `output_dir`, run `python scripts/repush_sp1.py --output-dir <that_dir> --hub-repo <repo>`
   (it will fall back to the latest `checkpoint-*` if `final_adapter/` is absent).
3. Optionally re-run `scripts/eval_sp1.py` to confirm the pushed merged model transcribes.

---

## Self-Review

- **Spec coverage:** §3 disk layout (Tasks 2/3/4 write final_adapter/merged under output_dir);
  §4 export refactor — save_adapter (T2), merge_to_dir (T3), push_folder (T1), save_and_push (T4),
  old functions removed (T5); §5 training flow (T5 cli wiring, push=not no_push, recovery print);
  §6 re-push script — pure planner (T6) + thin CLI (T7) + checkpoint fallback; §7 error handling
  (T4 failure-tolerance test, push_folder create_repo exist_ok, T6 missing-adapter error); §8
  testing — all CPU with mocks/fakes, pure planner injected predicates. All spec sections map to a
  task.
- **Placeholders:** none — every step ships runnable code/tests and exact commands.
- **Type consistency:** `save_adapter(model, processor, dir) -> dir`,
  `merge_to_dir(adapter_source, merged_dir, *, compute_dtype) -> merged_dir`,
  `push_folder(local_dir, repo_id, *, commit_message, private, allow_patterns)`,
  `save_and_push(...) -> {"adapter_dir","merged_dir","pushed","error"}`,
  `resolve_repush_plan(...) -> RepushPlan(adapter_source, merged_dir, need_merge, adapter_repo,
  merged_repo, compute_dtype)` are used consistently across export.py, cli.py, repush.py, and
  both CLIs.
```
