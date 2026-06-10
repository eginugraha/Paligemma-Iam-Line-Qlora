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

# PaliGemma-3B QLoRA — IAM-line Handwritten Text Recognition (LoRA adapter)

A **LoRA adapter** that fine-tunes [`google/paligemma-3b-pt-448`](https://huggingface.co/google/paligemma-3b-pt-448)
with **QLoRA** to transcribe single lines of **handwritten English** text. Trained on the
[`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line) dataset.

This repository contains **only the adapter** (a few tens of MB) — you load it on top of the
base PaliGemma at runtime. A self-contained merged model is published separately at
`eginugraha/paligemma-iam-line-qlora-merged`.

> Part of an undergraduate thesis comparing HTR scenarios (baseline, chain-of-thought, and a
> text-RAG spelling corrector). This adapter is the **M1 baseline** model.

## Results

Evaluated on the **IAM-line test split** (2,915 lines), base loaded in 4-bit + this adapter:

| Metric | Score |
|---|---|
| **CER** (Character Error Rate) | **17.37 %** |
| **WER** (Word Error Rate) | **28.34 %** |

(~21 % of lines transcribed perfectly; ~71 % under 25 % CER. Metrics computed with `jiwer`.)

## Usage

The base model is **gated** — accept the license on the
[base model page](https://huggingface.co/google/paligemma-3b-pt-448) and log in
(`huggingface-cli login`) first.

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

# Load the base in 4-bit (QLoRA config) — the lightest, ~4 GB VRAM, and the config the
# reported numbers were measured on. Use a CUDA GPU (bitsandbytes is CUDA-only).
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

# Transcribe one handwriting-line image.
image = Image.open("line.png").convert("RGB")   # IAM lines need 3-channel RGB
inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(model.device)
prompt_len = inputs["input_ids"].shape[1]       # remember the prompt length

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64)   # greedy; IAM lines are short

# Keep ONLY the generated continuation (drop the prompt prefix), then decode.
text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
print(text)
```

For full precision instead of 4-bit, drop `quantization_config=bnb` and pass
`torch_dtype=torch.bfloat16`.

## Training

QLoRA fine-tune of the frozen 4-bit base with a small LoRA adapter on the attention projections.

| Setting | Value |
|---|---|
| Base model | `google/paligemma-3b-pt-448` (image size 448) |
| Dataset | `Teklia/IAM-line` (train split) |
| Quantization | 4-bit **NF4**, double quant, compute dtype bf16 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 1 × grad-accum 8 (effective 8) |
| Max target tokens | 64 |
| Mixed precision | bf16 |
| Hardware | 1× **NVIDIA RTX A6000** (48 GB), ~4h11m |
| Prompt | `transcribe the handwritten text\n` |

The vision tower and multimodal projector are **not** adapted (attention-only LoRA) to keep the
adapter tiny and robust across `transformers` versions.

## Intended use & limitations

- **Intended:** transcribing single lines of English handwriting similar to IAM (research /
  thesis use). Use the exact training prompt above.
- **Limitations:** line-level only (not full pages/paragraphs); English only; trained on IAM's
  style, so out-of-distribution handwriting will degrade. A small tail of lines (~5 %) can show
  repetition collapse (e.g. `# # #`). The base is gated under the Gemma license.

## Citation

If you use this adapter, please credit the base model and the IAM-line dataset:

- PaliGemma — Google.
- IAM-line — Teklia (derived from the IAM Handwriting Database).
