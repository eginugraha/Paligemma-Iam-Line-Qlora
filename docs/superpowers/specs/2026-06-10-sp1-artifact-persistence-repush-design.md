# SP-1 — Artifact Persistence & Re-push — Design

**Date:** 2026-06-10
**Status:** Approved (design)
**Depends on:** SP-1 existing modules (`htr_sp1.export`, `htr_sp1.model`, `htr_sp1.train`, `htr_sp1.cli`).
**Motivation:** A previous run finished training but the Hub push failed, leaving no pushed
adapter/merged model. The merged model was never written to local disk (only pushed), so a
failed push risked losing the deliverable. This work makes training artifacts durable on disk
*before* any network push, and adds a standalone recovery script to re-push them.

---

## 1. Purpose

Two outcomes:

1. **Harden training** so the adapter and merged model are saved to local disk as a
   definition-of-done artifact **before** pushing to the Hub, and a push failure is non-fatal
   and recoverable (the run is never lost to a network error).
2. **Standalone re-push** (`scripts/repush_sp1.py`) that loads the saved adapter (or, as a
   fallback, the latest Trainer checkpoint) from disk and pushes the adapter + merged model to
   the Hub — no retraining required.

This directly fixes the "trained but not pushed" case: the artifacts are on disk first, and
re-pushing is a one-command recovery.

## 2. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Scope | Both: standalone re-push script **and** harden the training flow |
| Disk persistence | Save **adapter + merged** to disk every run (~6 GB/run) |
| Push mechanism | **Directory upload** via `huggingface_hub.HfApi.upload_folder` (separate merge from push); no model loaded into RAM to push |
| Overwrite behaviour | Push to the **same** repo (new commit overwrites prior version; Hub keeps history, reversible) |
| Recovery source | `--adapter` defaults to `<output_dir>/final_adapter`, falling back to the latest Trainer checkpoint (`train.find_resume_checkpoint`) — recovers a run saved with the old code |
| Merge precision | Merge on a CPU-loaded **full-precision** base (existing clean-merge logic), default `bfloat16` |

## 3. Disk layout

Under the run's `output_dir`:

```
<output_dir>/
  final_adapter/      # LoRA adapter weights + adapter_config + processor (~tens of MB)
  merged/             # merged fp16 model + processor (~5.8 GB)
  test_metrics.json   # already written by the eval step
  checkpoint-<step>/  # existing HF Trainer checkpoints (contain adapter weights too)
```

`final_adapter/` is a dedicated, unambiguous copy of the final adapter (distinct from the
Trainer `checkpoint-<step>/` dirs, which also contain adapter weights but with extra training
state). The fallback path lets recovery work even when `final_adapter/` is absent (old runs).

## 4. `export.py` refactor (separate save / merge / push)

`export.py` currently couples merge+push and only operates on the in-memory PEFT model
(`push_adapter`, `push_merged`). Decompose into single-purpose, individually testable units.
`adapter_repo_id` / `merged_repo_id` are unchanged. `push_adapter` / `push_merged` are
**replaced** by the functions below (callers updated accordingly).

- `save_adapter(model, processor, adapter_dir) -> str`
  `model.save_pretrained(adapter_dir)` (PEFT writes only the adapter) + `processor.save_pretrained(adapter_dir)`.
  Returns `adapter_dir`.

- `merge_to_dir(adapter_source, merged_dir, *, compute_dtype="bfloat16") -> str`
  The existing clean merge, but **adapter comes from a path/hub-id** and the result is **written
  to disk** instead of pushed: load the base at full precision on CPU (no quantization, no
  `device_map`), `PeftModel.from_pretrained(base, adapter_source).merge_and_unload()`, then
  `merged.save_pretrained(merged_dir)`. The processor is copied into `merged_dir` too (so the
  dir is self-contained). Returns `merged_dir`.

- `push_folder(local_dir, repo_id, *, commit_message=None, private=True) -> None`
  The single place a push happens. `HfApi().create_repo(repo_id, private=private, exist_ok=True,
  repo_type="model")` then `HfApi().upload_folder(folder_path=local_dir, repo_id=repo_id,
  repo_type="model", commit_message=commit_message)`. Uploading to an existing repo adds a
  commit (overwrites same-named files; history preserved).

- `save_and_push(model, processor, *, output_dir, hub_repo, compute_dtype="bfloat16", push=True) -> dict`
  Orchestrates the definition-of-done for a fresh run:
  1. `adapter_dir = save_adapter(model, processor, <output_dir>/final_adapter)`
  2. `merged_dir = merge_to_dir(adapter_dir, <output_dir>/merged, compute_dtype=compute_dtype)`
  3. when `push` is True, push both via `push_folder` to `adapter_repo_id(hub_repo)` /
     `merged_repo_id(hub_repo)`, wrapped so a push exception is caught (see §6).
  Returns a status dict: `{"adapter_dir", "merged_dir", "pushed": bool, "error": str | None}`.
  Steps 1–2 always run (artifacts durable first); step 3 is the only part allowed to fail.
  `push=False` (used by `--no-push`) still writes both dirs but skips the network entirely
  (`pushed=False`, `error=None`).

## 5. Training flow change (`cli.py`)

Replace step 6 of `main` (currently `push_adapter` + `push_merged` + reload-validation) with:

- Call `export.save_and_push(model, processor, output_dir=rc.output_dir, hub_repo=rc.hub_repo,
  compute_dtype="bfloat16" if rc.bf16 else "float16", push=not rc.no_push)`. With `--no-push`,
  adapter+merged are still written to disk but the push and reload-validation are skipped.
- If the returned status has `pushed=False`, print the exact recovery command
  (`python scripts/repush_sp1.py --output-dir <...> --hub-repo <...>`) and the local artifact
  paths; do **not** crash the run.
- Reload-validation runs only when `pushed=True`.

## 6. Re-push script (`scripts/repush_sp1.py`)

Thin CLI, mirrors `scripts/eval_sp1.py` structure (src on path; `.env`/`HF_TOKEN` already loaded
by `htr_sp1.config`). Arguments:

- `--output-dir` (default `$HTR_OUTPUT_DIR` / `config.OUTPUT_DIR`)
- `--hub-repo` (default `$HTR_HUB_REPO_ID` / `config.HF_HUB_REPO_ID`)
- `--adapter` (default `<output_dir>/final_adapter`; if missing, fall back to
  `train.find_resume_checkpoint(output_dir)`; error clearly if neither exists)
- `--compute-dtype` (default `bfloat16`)

Resolution logic is extracted into a **pure, testable** helper
`resolve_repush_plan(output_dir, hub_repo, adapter, dtype, *, exists) -> RepushPlan` that decides
the adapter source, whether `<output_dir>/merged` already exists (push as-is) or must be built
(`merge_to_dir`), and the target repo ids — without touching the model or network (`exists` is an
injected predicate so the test uses a temp dir).

`main` executes the plan: optionally `merge_to_dir`, then `push_folder` for adapter and merged,
printing what was pushed.

## 7. Error handling

- **Push failure during training:** caught in `save_and_push`; artifacts already on disk; run
  exits cleanly with recovery instructions. (The core fix.)
- **Existing repo:** `create_repo(exist_ok=True)` + `upload_folder` → new commit on the same
  repo, reversible via Hub revision history.
- **Merge safety:** unchanged from current code — full-precision base on CPU avoids GPU OOM and
  the 4-bit merge corruption.
- **Missing adapter on re-push:** explicit error naming both the expected `final_adapter` path
  and the checkpoint fallback that were tried.

## 8. Testing strategy (CPU, no GPU/network)

Follow existing `tests/test_export.py` and `tests/test_cli.py` style (fakes + mocks).

- `save_adapter` — fake model/processor record `save_pretrained(dir)` calls; assert paths.
- `merge_to_dir` — mock the heavy `transformers`/`peft` loads; assert it reads from the given
  adapter source and writes `save_pretrained(merged_dir)`; returns the dir.
- `push_folder` — mock `HfApi`; assert `create_repo(exist_ok=True)` then `upload_folder` with the
  right `folder_path`/`repo_id`.
- `save_and_push` — happy path (calls save→merge→push, `pushed=True`); push-failure path
  (`push_folder` raises → `pushed=False`, `error` set, no exception propagates, dirs still
  returned).
- `resolve_repush_plan` — pure logic with an injected `exists`: default adapter, checkpoint
  fallback, "merged exists → skip merge" vs "must build", repo-id derivation, missing-adapter
  error.

`PaliGemmaForConditionalGeneration` / real uploads are never exercised in CI; the real GPU merge
+ Hub round-trip is validated manually on the training server (as with training itself).

## 9. Out of scope

- Changing the training/eval logic itself (only the artifact-persistence + push tail changes).
- GGUF/MLX conversion of the merged model (separate, deferred sub-project).
- Versioned/separate repos per run, confirmation prompts (decided: overwrite same repo).
