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
