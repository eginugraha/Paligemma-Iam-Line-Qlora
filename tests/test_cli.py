"""Tests for the CLI orchestration logic of SP-1 (`scripts/train_sp1.py` lives on top of
`htr_sp1.cli`). We test the *pure* pieces — argument parsing, precision resolution, and the
config fallback chain — without touching a GPU. The heavy `main()` pipeline (model load +
training) is exercised on a real GPU, exactly like `run_training` itself.
"""
from htr_sp1 import cli, config


# --- precision resolution -------------------------------------------------------------

def test_resolve_precision_passthrough():
    # Explicit choices are returned unchanged.
    assert cli.resolve_precision("bf16") == "bf16"
    assert cli.resolve_precision("fp16") == "fp16"


def test_resolve_precision_auto_uses_detection():
    # "auto" defers to the runtime GPU probe (fp16 on the CPU-only test machine).
    assert cli.resolve_precision("auto") == config.detect_precision()


def test_precision_to_settings_maps_to_dtype_and_flag():
    # bf16 -> bfloat16 compute + bf16 training flag; fp16 -> float16 + no bf16.
    assert cli.precision_to_settings("bf16") == {"compute_dtype": "bfloat16", "bf16": True}
    assert cli.precision_to_settings("fp16") == {"compute_dtype": "float16", "bf16": False}


# --- argument parsing -----------------------------------------------------------------

def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.precision == "auto"
    assert args.epochs is None
    assert args.output_dir is None
    assert args.hub_repo is None
    assert args.batch_size is None
    assert args.skip_sanity is False
    assert args.no_push is False
    assert args.no_eval is False


def test_parser_flags():
    args = cli.build_parser().parse_args(
        ["--precision", "bf16", "--epochs", "5", "--output-dir", "/workspace/out",
         "--hub-repo", "me/repo", "--batch-size", "4",
         "--skip-sanity", "--no-push", "--no-eval"]
    )
    assert args.precision == "bf16"
    assert args.epochs == 5
    assert args.output_dir == "/workspace/out"
    assert args.hub_repo == "me/repo"
    assert args.batch_size == 4
    assert args.skip_sanity is True
    assert args.no_push is True
    assert args.no_eval is True


# --- config fallback chain: CLI arg > env var > config default ------------------------

def test_resolve_config_cli_wins():
    args = cli.build_parser().parse_args(
        ["--output-dir", "/cli/out", "--hub-repo", "cli/repo", "--epochs", "7"]
    )
    env = {"HTR_OUTPUT_DIR": "/env/out", "HTR_HUB_REPO_ID": "env/repo"}
    rc = cli.resolve_config(args, env=env)
    assert rc.output_dir == "/cli/out"
    assert rc.hub_repo == "cli/repo"
    assert rc.epochs == 7


def test_resolve_config_env_then_default():
    args = cli.build_parser().parse_args([])  # nothing on the CLI
    env = {"HTR_OUTPUT_DIR": "/env/out", "HTR_HUB_REPO_ID": "env/repo"}
    rc = cli.resolve_config(args, env=env)
    # No CLI value -> env var is used.
    assert rc.output_dir == "/env/out"
    assert rc.hub_repo == "env/repo"
    # No CLI epochs and no env for it -> the config.py default.
    assert rc.epochs == config.NUM_TRAIN_EPOCHS


def test_resolve_config_falls_back_to_config_defaults():
    args = cli.build_parser().parse_args([])
    rc = cli.resolve_config(args, env={})  # neither CLI nor env set
    assert rc.output_dir == config.OUTPUT_DIR
    assert rc.hub_repo == config.HF_HUB_REPO_ID
    assert rc.batch_size == config.PER_DEVICE_TRAIN_BATCH_SIZE


def test_resolve_config_resolves_precision_and_settings():
    args = cli.build_parser().parse_args(["--precision", "bf16"])
    rc = cli.resolve_config(args, env={})
    assert rc.precision == "bf16"
    assert rc.compute_dtype == "bfloat16"
    assert rc.bf16 is True
