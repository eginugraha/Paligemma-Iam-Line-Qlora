# SP-1 — PaliGemma QLoRA for IAM-line Transcription

Fine-tunes `google/paligemma-3b-pt-448` with QLoRA on `Teklia/IAM-line` to transcribe
handwriting lines. Produced for the HTR thesis (see `docs/superpowers/specs/2026-06-08-sp1-model-training-design.md`).

## Layout
- `src/htr_sp1/` — tested, documented modules (config, data, metrics, model, inference, train, evaluate, export, repush, cli)
- `scripts/` — server entry points: `train_sp1.py`, `eval_sp1.py`, `repush_sp1.py`
- `notebooks/sp1_train.ipynb` — Colab orchestration (run on a T4)
- `tests/` — laptop-runnable unit tests (no GPU/downloads)
- Credentials/config (`HF_TOKEN`, `HTR_HUB_REPO_ID`, `HTR_OUTPUT_DIR`) can live in a gitignored
  `.env` at the repo root (see `.env.example`); `config.py` auto-loads it. Shell exports still win.

## Run the tests locally (no GPU)
```bash
pip install pytest jiwer
python -m pytest
```

## Train in Colab
1. Upload `src/`, `requirements.txt` to the Colab session (or clone the repo).
2. Open `notebooks/sp1_train.ipynb`, set `HTR_HUB_REPO_ID` to your HF repo.
3. Run top to bottom. The **sanity gate** (overfit 2 samples) must show loss→~0 before the full run.
4. Outputs: test CER/WER (`test_metrics.json`); the adapter + merged model are written to disk
   under the output dir (`final_adapter/`, `merged/`) **then** pushed to the Hub.

## Train on a server / CLI (e.g. RunPod A6000 Pod) — no notebook, no Drive
Same pipeline as the notebook, runnable over SSH. Persistence comes from the machine's own
disk (e.g. a RunPod Network Volume at `/workspace`), so there is **no Colab/Drive mounting**.

```bash
git clone <repository> /workspace/htr && cd /workspace/htr
pip install -r requirements.txt
export HTR_OUTPUT_DIR="/workspace/outputs"          # persistent dir for checkpoints
export HTR_HUB_REPO_ID="HF_ID/paligemma-iam-line-qlora"
huggingface-cli login                               # token with write access

python scripts/train_sp1.py                         # full pipeline; precision auto-detected
python scripts/train_sp1.py --help                  # all flags
nohup python scripts/train_sp1.py > train.log 2>&1 &   # headless; tail -f train.log
python scripts/eval_sp1.py --base-precision bf16 --limit 50 # smoke test 50 sample
python scripts/eval_sp1.py --base-precision 4bit --out test_metrics_4bit.json # eval with base precision 4 bit
python scripts/eval_sp1.py --base-precision bf16 --out test_metrics_bf16.json # eval with base precision 16 bit

huggingface-cli upload eginugraha/htr-sp1-run-artifacts /workspace/outputs/test_metrics.json test_metrics.json --repo-type dataset
huggingface-cli upload eginugraha/htr-sp1-run-artifacts /workspace/htr/train.log train.log --repo-type dataset
```
ps aux | grep python scripts/train_sp1.py
- **Precision:** `--precision auto` (default) picks **bf16** on Ampere/Ada GPUs (A6000/3090/4090), **fp16** on a T4 — no code edit needed when moving machines.
- **Config precedence:** CLI flag > env var (`HTR_*`) > `config.py` default. Override per run with `--epochs`, `--batch-size`, `--output-dir`, `--hub-repo`.
- **Skip flags:** `--skip-sanity`, `--no-eval`, `--no-push` (the last writes artifacts to disk but
  skips the Hub upload + reload-validation).
- Resumes automatically from the latest checkpoint in `--output-dir` if interrupted.

## Artifacts & recovery (re-push without retraining)
Every run writes its artifacts to disk **before** pushing, so a failed push never costs a retrain:
- `<output_dir>/final_adapter/` — the LoRA adapter (+ processor)
- `<output_dir>/merged/` — the self-contained merged fp16 model

If the push failed (token/network) or you want to re-publish, re-push from disk — no retraining:
```bash
python scripts/repush_sp1.py --output-dir /workspace/outputs --hub-repo eginugraha/paligemma-iam-line-qlora
```
It reuses `<output_dir>/merged/` if present (else rebuilds it) and falls back to the latest
`checkpoint-*` when `final_adapter/` is absent (older runs). All branching is in the unit-tested
`htr_sp1.repush.resolve_repush_plan`.

## Inference interface (the contract SP-2 imports)
```python
from htr_sp1.inference import generate_transcription
# model, processor: a loaded PaliGemma (base+adapter, or the merged repo) + its processor
text = generate_transcription(model, processor, pil_image)
```
- **Input:** loaded model, processor, one `PIL.Image` of a handwriting line.
- **Output:** predicted transcription `str` (whitespace-stripped).
- **M1** in SP-2 = this call. **M2 (CoT)** = same model, different prompt (SP-2 swaps the prompt).

## Definition of Done
- Full QLoRA run completed; val-CER stabilized.
- Test CER/WER reported.
- Adapter + merged saved to disk (`final_adapter/`, `merged/`), then pushed to the Hub (private);
  a failed push is recoverable from disk via `scripts/repush_sp1.py` (no retrain).
- Fresh reload from the Hub transcribes sample images correctly (validation gate).
