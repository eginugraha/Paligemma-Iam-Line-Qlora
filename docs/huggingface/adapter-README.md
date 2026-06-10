---
license: gemma
base_model: google/paligemma-3b-pt-448
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - lora
  - qlora
  - peft
  - paligemma
  - handwritten-text-recognition
  - htr
  - ocr
  - vision-language
datasets:
  - Teklia/IAM-line
language:
  - en
metrics:
  - cer
  - wer
---

# Model Card for PaliGemma-3B QLoRA — IAM-line Handwritten Text Recognition (LoRA adapter)

A **QLoRA LoRA adapter** for `google/paligemma-3b-pt-448` that transcribes single lines of
**handwritten English** text, fine-tuned on the `Teklia/IAM-line` dataset. This repo holds the
**adapter only** (a few tens of MB); load it on top of the base model. A self-contained merged
version is published at `eginugraha/paligemma-iam-line-qlora-merged`.

## Model Details

### Model Description

- **Developed by:** Putu Bagus Indra Dermawan Kemuning, Lawy Xenna L. Gaol, and Egi Nugraha — undergraduate thesis project (Hugging Face repo owner: [`eginugraha`](https://huggingface.co/eginugraha))
- **Funded by:** Not externally funded (self-funded undergraduate thesis)
- **Shared by:** Putu Bagus Indra Dermawan Kemuning, Lawy Xenna L. Gaol, Egi Nugraha
- **Model type:** LoRA adapter (PEFT) for a vision-language model — image-line → text (handwritten text recognition)
- **Language(s) (NLP):** English (`en`)
- **License:** `gemma` (inherited from the base model's Gemma license)
- **Finetuned from model:** [`google/paligemma-3b-pt-448`](https://huggingface.co/google/paligemma-3b-pt-448)

### Model Sources

- **Repository:** GitHub — [`https://github.com/eginugraha/Paligemma-Iam-Line-Qlora`](https://github.com/eginugraha/Paligemma-Iam-Line-Qlora)
- **Paper:** Undergraduate thesis (in progress; not yet published)
- **Demo:** N/A

## Uses

### Direct Use

Transcribe a single line of English handwriting (IAM-style) to text. Use the exact training
prompt `transcribe the handwritten text\n`. Suitable for research and thesis reproduction.

### Downstream Use

Serves as the **M1 baseline** model in a larger HTR comparison: its output is consumed by a
FastAPI backend (SP-2) and optionally repaired by a text-RAG spelling corrector (SP-3, scenarios
M3/M4). The adapter can also be merged into the base for single-artifact deployment.

### Out-of-Scope Use

- Full-page / multi-line / paragraph documents (this is **line-level** only).
- Languages other than English; printed/typed text; non-IAM handwriting styles (expect degradation).
- Any safety-critical or high-stakes transcription without human review.

## Bias, Risks, and Limitations

- Trained on IAM (mostly modern English handwriting from a limited writer pool), so it is biased
  toward that distribution and degrades on out-of-distribution scripts/styles.
- Line-level only; no layout/segmentation.
- A small tail of lines (~5%) can exhibit **repetition collapse** (e.g. `# # #`, `" " "`).
- The base model is gated under the Gemma license; its terms apply.

### Recommendations

Users (both direct and downstream) should be aware of the limitations above. Feed it
pre-segmented single lines, use the exact prompt, and review outputs for critical use. Optional
generation guards (`no_repeat_ngram_size`, `repetition_penalty`) can reduce the repetition tail.

## How to Get Started with the Model

The base model is **gated** — accept its license and `huggingface-cli login` first (or set `HF_TOKEN`).

```python
import torch
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from peft import PeftModel
from PIL import Image

BASE = "google/paligemma-3b-pt-448"
ADAPTER = "eginugraha/paligemma-iam-line-qlora-adapter"
PROMPT = "transcribe the handwritten text\n"   # the exact training prompt

# Load the base in 4-bit (QLoRA config) — lightest (~4 GB VRAM) and the config the reported
# numbers were measured on. Requires a CUDA GPU (bitsandbytes is CUDA-only).
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
processor = PaliGemmaProcessor.from_pretrained(BASE)
base = PaliGemmaForConditionalGeneration.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER).eval()

image = Image.open("line.png").convert("RGB")   # IAM lines need 3-channel RGB
inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(model.device)
prompt_len = inputs["input_ids"].shape[1]       # remember prompt length to slice it off

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64)   # greedy; IAM lines are short

text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
print(text)
```

For full precision instead of 4-bit, drop `quantization_config=bnb` and pass `torch_dtype=torch.bfloat16`.

## Training Details

### Training Data

[`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line) — **train split only**
(line-level images + transcriptions from the IAM Handwriting Database).

### Training Procedure

#### Preprocessing

- Images converted to 3-channel RGB; resized to 448×448 by the PaliGemma processor.
- Each example encoded by the processor as prompt (`transcribe the handwritten text\n`) + image,
  with the ground-truth transcription passed as the `suffix` (prompt tokens are loss-masked).

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision over a **4-bit NF4** (double-quant) frozen base (QLoRA)
- **LoRA:** rank 8, alpha 16, dropout 0.05; target modules `q_proj, k_proj, v_proj, o_proj` (attention only)
- **Optimizer LR:** 2e-4
- **Epochs:** 3 (2,430 steps)
- **Batch size:** 1 per device × gradient accumulation 8 (effective 8)
- **Max target tokens:** 64
- **Seed:** fixed for reproducibility

#### Speeds, Sizes, Times

- ~4 h 11 m wall-clock on a single GPU; final train loss ≈ 0.64, eval loss ≈ 0.75.
- Adapter size: a few tens of MB (LoRA weights only; the ~2.92B-parameter base is not stored here).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

`Teklia/IAM-line` **test split** (2,915 lines), unseen during training.

#### Factors

Evaluated overall on the test split (no per-writer/style breakdown). Observed: ~21% of lines
transcribed perfectly; ~71% under 25% CER; a ~5% tail above 50% CER (repetition collapse).

#### Metrics

- **CER (Character Error Rate)** and **WER (Word Error Rate)** via `jiwer` — the standard
  line-level HTR metrics (lower is better).

### Results

| Metric | Score |
|---|---|
| **CER** | **17.37 %** |
| **WER** | **28.34 %** |

Measured with the base loaded in **4-bit + adapter** (the unmerged inference path).

#### Summary

A healthy, reportable first QLoRA baseline for line-level English HTR on IAM.

## Model Examination

Error profile is dominated by minor character substitutions; a small tail shows repetition
collapse on hard/ambiguous lines, mitigable with repetition-control generation flags.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) (Lacoste et al., 2019).

- **Hardware Type:** 1× NVIDIA RTX A6000 (48 GB)
- **Hours used:** ~4.2
- **Cloud Provider:** Runpod.io
- **Compute Region:** Not recorded
- **Carbon Emitted:** Not measured

## Technical Specifications

### Model Architecture and Objective

Base: PaliGemma-3B (SigLIP vision encoder + Gemma-2B language decoder + linear projector). This
adapter adds low-rank updates to the decoder's attention projections only. Objective:
autoregressive (causal-LM) next-token prediction of the transcription conditioned on the image
and prompt.

### Compute Infrastructure

#### Hardware

Single NVIDIA RTX A6000 (48 GB). (Note: a 24 GB RTX A5000 ran out of memory at the first
end-of-epoch evaluation over PaliGemma's ~257k-token vocabulary, which is why a 48 GB card was used.)

#### Software

Python; `torch==2.3.1`, `transformers==4.42.4`, `peft==0.11.1`, `bitsandbytes==0.43.1`,
`accelerate==0.31.0`, `datasets==2.20.0`, `jiwer==3.0.4`.

## Citation

**BibTeX:**

```bibtex
@misc{kemuning_paligemma_iam_line_qlora,
  author = {Kemuning, Putu Bagus Indra Dermawan and Gaol, Lawy Xenna L. and Nugraha, Egi},
  title  = {PaliGemma-3B QLoRA adapter for IAM-line handwritten text recognition},
  year   = {2026},
  note   = {Undergraduate thesis project},
  howpublished = {Hugging Face Hub: eginugraha/paligemma-iam-line-qlora-adapter}
}
```

**APA:**

Kemuning, P. B. I. D., Gaol, L. X. L., & Nugraha, E. (2026). *PaliGemma-3B QLoRA adapter for IAM-line handwritten text recognition* [LoRA adapter]. Hugging Face. https://huggingface.co/eginugraha/paligemma-iam-line-qlora-adapter

Please also credit the base model (PaliGemma, Google) and the dataset (IAM-line, Teklia; derived from the IAM Handwriting Database).

## Glossary

- **CER** — Character Error Rate: edit distance between prediction and reference, normalized by reference length.
- **WER** — Word Error Rate: the same at word granularity.
- **QLoRA** — fine-tuning a 4-bit-quantized base with a small trainable LoRA adapter.
- **LoRA** — Low-Rank Adaptation: trains small rank-decomposition matrices instead of full weights.

## More Information

This adapter is the M1 baseline of a four-scenario HTR thesis (baseline, chain-of-thought, and a
text-RAG lexical corrector). See the project repository for the full pipeline and evaluation code.

## Model Card Authors

Putu Bagus Indra Dermawan Kemuning, Lawy Xenna L. Gaol, and Egi Nugraha (Hugging Face: [`eginugraha`](https://huggingface.co/eginugraha))

## Model Card Contact

Via the Hugging Face profile [`eginugraha`](https://huggingface.co/eginugraha).
