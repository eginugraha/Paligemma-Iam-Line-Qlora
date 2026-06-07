"""Test the TrainingArguments assembly — the knobs that make training fit a T4 and survive
Colab disconnects. The real fit() runs in the notebook.
"""
from htr_sp1 import config, train


def test_training_args_fit_t4_and_checkpoint():
    args = train.build_training_args(output_dir="/tmp/run")
    assert args["output_dir"] == "/tmp/run"
    assert args["per_device_train_batch_size"] == config.PER_DEVICE_TRAIN_BATCH_SIZE
    assert args["gradient_accumulation_steps"] == config.GRAD_ACCUMULATION_STEPS
    # Gradient checkpointing trades compute for memory — required on 16GB.
    assert args["gradient_checkpointing"] is True
    # We must save checkpoints so a Colab disconnect can resume (not lose hours of training).
    assert args["save_strategy"] == "epoch"
    # Evaluate each epoch so we can watch val-CER and stop when it stabilizes.
    assert args["evaluation_strategy"] == "epoch"
    # T4 (Turing) supports fp16, not bf16 — assert the documented hardware choice.
    assert args["fp16"] is True
    # No external experiment trackers — keep the thesis run self-contained.
    assert args["report_to"] == "none"


def test_find_resume_checkpoint_prefers_existing(tmp_path):
    # No checkpoints yet -> None (start fresh).
    assert train.find_resume_checkpoint(str(tmp_path)) is None
    # Create a checkpoint dir -> it should be returned so training resumes from it.
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    assert train.find_resume_checkpoint(str(tmp_path)) == str(ckpt)
    # A LOWER step number created later must NOT win — guards against lexicographic sort
    # (where "checkpoint-25" would wrongly beat "checkpoint-100").
    (tmp_path / "checkpoint-25").mkdir()
    assert train.find_resume_checkpoint(str(tmp_path)) == str(ckpt)
