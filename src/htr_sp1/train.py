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


def build_training_args(output_dir: str) -> Dict[str, Any]:
    """Return TrainingArguments fields tuned for a Colab T4 with frequent checkpoints.

    Args:
        output_dir: Where checkpoints/logs are written (point at Drive in Colab).
    """
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRAD_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "num_train_epochs": config.NUM_TRAIN_EPOCHS,
        "gradient_checkpointing": True,   # memory-for-compute trade; needed on 16GB
        "fp16": True,                     # T4 (Turing) supports fp16, not bf16
        "save_strategy": "epoch",         # checkpoint every epoch -> resumable
        "evaluation_strategy": "epoch",   # val metrics every epoch -> watch val-CER
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


def run_training(model, processor, train_ds, eval_ds, output_dir: str):
    """Run the real fine-tuning loop on Colab/GPU, resuming if a checkpoint exists.

    Heavy imports are local so the module stays laptop-importable for unit tests.

    Args:
        model: PEFT-wrapped PaliGemma from `model.load_trainable_model`.
        processor: The matching processor.
        train_ds / eval_ds: HF datasets yielding {"image", "text"} records.
        output_dir: Checkpoint/log directory (Drive path in Colab).

    Returns:
        The trained model (LoRA weights updated in place).
    """
    from transformers import Trainer, TrainingArguments

    from .data import build_training_example

    def collate(batch):
        # Encode each record (image + prompt + label suffix) and let the processor build
        # the padded tensors + masked labels. One example per record keeps memory predictable.
        examples = [build_training_example(r, processor) for r in batch]
        return processor.tokenizer.pad(examples, return_tensors="pt") if hasattr(processor, "tokenizer") else examples[0]

    args = TrainingArguments(**build_training_args(output_dir))
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
