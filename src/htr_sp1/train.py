"""Training orchestration.

Split into testable pieces:
  - `build_training_args` returns a plain dict of HF TrainingArguments fields (T4-friendly,
    checkpoint-on-epoch) that we can assert on without importing transformers.
  - `find_resume_checkpoint` scans the output dir so a Colab disconnect resumes instead of
    restarting — central to the "survive disconnects" requirement.
  - `run_training` wires everything to a real Trainer (Colab/GPU only).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from . import config


def build_training_args(output_dir: str, *, bf16: bool = False) -> Dict[str, Any]:
    """Return TrainingArguments fields tuned for a T4-class run with frequent checkpoints.

    Mixed precision: HF errors if both fp16 and bf16 are enabled, so we set exactly one. The
    default is fp16 (the T4-safe baseline); pass bf16=True on an Ampere/Ada GPU for faster,
    more stable training.

    Args:
        output_dir: Where checkpoints/logs are written (a persistent volume on a server).
        bf16: Use bfloat16 mixed precision instead of float16.
    """
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        # MUST set this explicitly. HF defaults per_device_eval_batch_size to 8 when omitted, and
        # an eval batch of 8 over PaliGemma's ~257k vocab allocates ~8 GiB for the cross-entropy
        # logits in one shot -> CUDA OOM at the first end-of-epoch eval on a 24 GB GPU (that OOM
        # is exactly why training moved to a 48 GB A6000). Keep eval at 1.
        "per_device_eval_batch_size": config.PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRAD_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "num_train_epochs": config.NUM_TRAIN_EPOCHS,
        # Keep the raw {image, text} columns so our collate() can encode them. Without this,
        # Trainer strips columns not in the model's forward signature BEFORE collation, leaving
        # empty batches ("No columns ... match the model's forward method signature").
        "remove_unused_columns": False,
        "gradient_checkpointing": True,   # memory-for-compute trade; needed on 16GB
        # Exactly one mixed-precision mode on — never both (HF raises if both are True).
        "fp16": not bf16,                 # T4 (Turing) supports fp16, not bf16
        "bf16": bf16,                     # Ampere/Ada (e.g. A6000) supports bf16
        "save_strategy": "epoch",         # checkpoint every epoch -> resumable
        "evaluation_strategy": "epoch",   # eval each epoch; live signal is eval LOSS (Trainer
        # does not generate during eval), while true val/test CER is computed via evaluate_split.
        "logging_steps": 25,
        "report_to": "none",              # no external trackers; keep the thesis run simple
    }


def find_resume_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path of the latest checkpoint to resume from, or None to start fresh.

    HF Trainer writes folders named `checkpoint-<step>`. After a Colab disconnect we want to
    pick up where we left off rather than retrain from scratch.

    Args:
        output_dir: The training output directory (possibly on Drive).
    """
    if not os.path.isdir(output_dir):
        return None
    # Collect checkpoint dirs and choose the one with the highest step number.
    checkpoints = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not checkpoints:
        return None
    # Sort by the trailing integer step so "checkpoint-100" beats "checkpoint-25".
    return max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[-1]))


def collate_examples(batch, processor):
    """Turn a batch of {image, text} records into padded PaliGemma training inputs.

    We call the processor ONCE on the whole batch — lists of prompts, RGB images, and the
    ground-truth texts as `suffix`. The processor then builds batched, padded input_ids /
    attention_mask / pixel_values AND the loss labels (prompt tokens masked, suffix kept)
    natively.

    Why not encode per-record then `tokenizer.pad`? That path leaves a leading [1, seq_len]
    dim on every tensor (-> [batch, 1, seq_len] after stacking) and does NOT pad/stack the
    image `pixel_values` at all. The model then receives 4-D hidden states and crashes with
    "too many values to unpack (expected 3)". Batched processing is the idiomatic, correct fix.

    Args:
        batch: A list of records, each with "image" (PIL) and "text" (ground truth).
        processor: A PaliGemmaProcessor (or compatible fake in tests).
    """
    from .data import build_prompt, ensure_rgb

    images = [ensure_rgb(record["image"]) for record in batch]
    prompts = [build_prompt() for _ in batch]
    suffixes = [record["text"] for record in batch]
    return processor(
        text=prompts,
        images=images,
        suffix=suffixes,
        return_tensors="pt",
        padding="longest",
    )


def run_training(model, processor, train_ds, eval_ds, output_dir: str, *, bf16: bool = False):
    """Run the real fine-tuning loop on GPU, resuming if a checkpoint exists.

    Heavy imports are local so the module stays laptop-importable for unit tests.

    Args:
        model: PEFT-wrapped PaliGemma from `model.load_trainable_model`.
        processor: The matching processor.
        train_ds / eval_ds: HF datasets yielding {"image", "text"} records.
        output_dir: Checkpoint/log directory (a persistent volume path on a server).
        bf16: Use bfloat16 mixed precision (Ampere/Ada); defaults to fp16.

    Returns:
        The trained model (LoRA weights updated in place).
    """
    from transformers import Trainer, TrainingArguments

    # Batched collation through the processor (see collate_examples for why per-record +
    # tokenizer.pad is wrong). Bound to this run's processor.
    def collate(batch):
        return collate_examples(batch, processor)

    args = TrainingArguments(**build_training_args(output_dir, bf16=bf16))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )
    # Resume from the latest checkpoint if Colab disconnected mid-run; else start fresh.
    trainer.train(resume_from_checkpoint=find_resume_checkpoint(output_dir))
    return model
