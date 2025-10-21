"""Benchmark supervised training with and without pretraining warm starts."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from market_nn_plus_ultra.training import (
    TrainerOverrides,
    load_experiment_from_file,
    run_training,
)


def _disable_wandb(config) -> None:
    config.wandb_project = None
    config.wandb_entity = None
    config.wandb_run_name = None
    config.wandb_tags = ()
    config.wandb_offline = True


def _build_overrides(args: argparse.Namespace) -> TrainerOverrides | None:
    overrides = TrainerOverrides(
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        batch_size=args.batch_size,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
    )
    return None if overrides.is_empty() else overrides


def _result_to_row(label: str, result, duration: float) -> dict[str, object]:
    row: dict[str, object] = {
        "run": label,
        "duration_seconds": duration,
        "best_model_path": result.best_model_path,
    }
    for key, value in result.logged_metrics.items():
        row[f"metric_{key.replace('/', '_')}"] = value
    for key, value in result.dataset_summary.items():
        row[f"dataset_{key}"] = value
    for key, value in result.profitability_summary.items():
        row[f"profitability_{key.replace('/', '_')}"] = value
    if result.profitability_reports:
        for kind, path in result.profitability_reports.items():
            row[f"profitability_report_{kind}"] = path
    return row


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Experiment YAML path")
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by the pretraining loop",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/pretraining_comparison.parquet"),
        help="Destination parquet path for aggregated results",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Optional directory to store scratch and warm-start checkpoints",
    )
    parser.add_argument("--label-scratch", type=str, default="scratch", help="Label used for the baseline run")
    parser.add_argument(
        "--label-pretrained",
        type=str,
        default="pretrained",
        help="Label used for the warm-start run",
    )
    parser.add_argument("--max-epochs", type=int, help="Override trainer max epochs for both runs")
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        help="Limit the fraction/number of training batches for rapid sweeps",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        help="Limit the fraction/number of validation batches for rapid sweeps",
    )
    parser.add_argument("--batch-size", type=int, help="Override the batch size for both runs")
    parser.add_argument("--accelerator", type=str, help="Override trainer accelerator (cpu/gpu/mps)")
    parser.add_argument(
        "--devices",
        type=str,
        help="Override trainer devices (int, 'auto', or accelerator-specific identifier)",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        help="Override logging frequency to keep console output manageable",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Keep Weights & Biases logging enabled (disabled by default)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _prepare_config(base_config, *, enable_wandb: bool, overrides: TrainerOverrides | None, checkpoint_dir: Path):
    config = copy.deepcopy(base_config)
    if not enable_wandb:
        _disable_wandb(config)
    if overrides is not None:
        overrides.apply(config.trainer)
    config.trainer.checkpoint_dir = checkpoint_dir
    return config


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    base_config = load_experiment_from_file(args.config)
    overrides = _build_overrides(args)

    output_dir = args.output.parent
    workdir = args.workdir or (output_dir / "pretraining_runs")
    scratch_dir = workdir / "scratch"
    warm_dir = workdir / "pretrained"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    warm_dir.mkdir(parents=True, exist_ok=True)

    scratch_config = _prepare_config(
        base_config,
        enable_wandb=args.enable_wandb,
        overrides=overrides,
        checkpoint_dir=scratch_dir,
    )
    warm_config = _prepare_config(
        base_config,
        enable_wandb=args.enable_wandb,
        overrides=overrides,
        checkpoint_dir=warm_dir,
    )

    rows: list[dict[str, object]] = []

    print("Running scratch baseline…")
    start = time.perf_counter()
    scratch_result = run_training(scratch_config)
    scratch_duration = time.perf_counter() - start
    rows.append(_result_to_row(args.label_scratch, scratch_result, scratch_duration))
    print(f"Scratch run completed in {scratch_duration:.2f}s")

    print("Running pretraining warm-start…")
    start = time.perf_counter()
    warm_result = run_training(
        warm_config,
        pretrain_checkpoint_path=args.pretrain_checkpoint,
    )
    warm_duration = time.perf_counter() - start
    rows.append(_result_to_row(args.label_pretrained, warm_result, warm_duration))
    print(f"Warm-start run completed in {warm_duration:.2f}s")

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output)
    print(f"Wrote {len(frame)} benchmark rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

