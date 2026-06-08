# SP-1 — PaliGemma QLoRA for IAM-line Transcription

Fine-tunes `google/paligemma-3b-pt-448` with QLoRA on `Teklia/IAM-line` to transcribe
handwriting lines. Produced for the HTR thesis (see `docs/superpowers/specs/2026-06-08-sp1-model-training-design.md`).

## Layout
- `src/htr_sp1/` — tested, documented modules (config, data, metrics, model, inference, train, evaluate, export)
- `notebooks/sp1_train.ipynb` — Colab orchestration (run on a T4)
- `tests/` — laptop-runnable unit tests (no GPU/downloads)

## Run the tests locally (no GPU)
```bash
pip install pytest jiwer
python -m pytest
```

## Train in Colab
1. Upload `src/`, `requirements.txt` to the Colab session (or clone the repo).
2. Open `notebooks/sp1_train.ipynb`, set `HTR_HUB_REPO_ID` to your HF repo.
3. Run top to bottom. The **sanity gate** (overfit 2 samples) must show loss→~0 before the full run.
4. Outputs: test CER/WER (`test_metrics.json` on Drive) + adapter & merged repos on the Hub.

## Train on a server / CLI (e.g. RunPod A5000 Pod) — no notebook, no Drive
Same pipeline as the notebook, runnable over SSH. Persistence comes from the machine's own
disk (e.g. a RunPod Network Volume at `/workspace`), so there is **no Colab/Drive mounting**.

```bash
git clone https://github.com/eginugraha/Paligemma-Iam-Line-Qlora.git /workspace/htr && cd /workspace/htr
pip install -r requirements.txt
export HTR_OUTPUT_DIR="/workspace/outputs"          # persistent dir for checkpoints
export HTR_HUB_REPO_ID="eginugraha/paligemma-iam-line-qlora"
huggingface-cli login                               # token with write access

python scripts/train_sp1.py                         # full pipeline; precision auto-detected
python scripts/train_sp1.py --help                  # all flags
nohup python scripts/train_sp1.py > train.log 2>&1 &   # headless; tail -f train.log
```
- **Precision:** `--precision auto` (default) picks **bf16** on Ampere/Ada GPUs (A5000/3090/4090), **fp16** on a T4 — no code edit needed when moving machines.
- **Config precedence:** CLI flag > env var (`HTR_*`) > `config.py` default. Override per run with `--epochs`, `--batch-size`, `--output-dir`, `--hub-repo`.
- **Skip flags:** `--skip-sanity`, `--no-eval`, `--no-push` (the last also skips reload-validation).
- Resumes automatically from the latest checkpoint in `--output-dir` if interrupted.

## Inference interface (the contract SP-2 imports)
```python
from htr_sp1.inference import generate_transcription
# model, processor: a loaded PaliGemma (merged repo) + its processor
text = generate_transcription(model, processor, pil_image)
```
- **Input:** loaded model, processor, one `PIL.Image` of a handwriting line.
- **Output:** predicted transcription `str` (whitespace-stripped).
- **M1** in SP-2 = this call. **M2 (CoT)** = same model, different prompt (SP-2 swaps the prompt).

## Definition of Done
- Full QLoRA run completed; val-CER stabilized.
- Test CER/WER reported.
- Adapter + merged pushed to the Hub (private).
- Fresh reload from the Hub transcribes sample images correctly (validation gate).
