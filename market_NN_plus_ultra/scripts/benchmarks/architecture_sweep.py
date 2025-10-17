"""CLI utility to benchmark multiple Plus Ultra architectures in sequence."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from market_nn_plus_ultra.training import (
    TrainerOverrides,
    flatten_benchmark_result,
    iter_scenarios,
    load_experiment_from_file,
    prepare_config_for_scenario,
    run_training,
)


def _parse_str_list(value: str | None, *, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [item.lower() for item in items] if items else list(default)


def _parse_int_list(value: str | None, *, default: Sequence[int]) -> list[int]:
    if value is None:
        return list(default)
    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            items.append(int(chunk))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid integer value: '{chunk}'") from exc
    return items or list(default)


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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Experiment YAML path")
    parser.add_argument(
        "--architectures",
        type=str,
        help="Comma separated list of architectures to benchmark (defaults to config value)",
    )
    parser.add_argument(
        "--model-dims",
        type=str,
        help="Comma separated list of model dimensions to evaluate (defaults to config value)",
    )
    parser.add_argument(
        "--depths",
        type=str,
        help="Comma separated list of backbone depths to evaluate (defaults to config value)",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        help="Comma separated list of forecast horizons to evaluate (defaults to config value)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override trainer max epochs for every sweep run",
    )
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
    parser.add_argument("--batch-size", type=int, help="Override the batch size for every run")
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
        "--label-template",
        type=str,
        help="Optional format string to name scenarios (use {architecture}, {model_dim}, {depth}, {horizon})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/architecture_sweep.parquet"),
        help="Destination parquet path for aggregated results",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Keep Weights & Biases logging enabled during sweeps (disabled by default)",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Print metric dictionaries for each run in addition to the summary line",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    base_config = load_experiment_from_file(args.config)
    default_architecture = [base_config.model.architecture]
    default_model_dim = [base_config.model.model_dim]
    default_depth = [base_config.model.depth]
    default_horizon = [base_config.model.horizon]

    architectures = _parse_str_list(args.architectures, default=default_architecture)
    model_dims = _parse_int_list(args.model_dims, default=default_model_dim)
    depths = _parse_int_list(args.depths, default=default_depth)
    horizons = _parse_int_list(args.horizons, default=default_horizon)

    overrides = _build_overrides(args)
    scenarios = list(
        iter_scenarios(
            architectures,
            model_dims,
            depths,
            horizons,
            label_template=args.label_template,
        )
    )

    if not scenarios:
        print("No scenarios produced from the provided search space.")
        return 1

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        config = prepare_config_for_scenario(
            base_config,
            scenario,
            overrides=overrides,
            disable_wandb=not args.enable_wandb,
        )
        print(
            f"Running {scenario.label or scenario.architecture} "
            f"(dim={scenario.model_dim}, depth={scenario.depth}, horizon={scenario.horizon})"
        )
        start = time.perf_counter()
        run_result = run_training(config)
        duration = time.perf_counter() - start
        row = flatten_benchmark_result(scenario, run_result, duration_seconds=duration)
        rows.append(row)
        val_loss = row.get("metric_val_loss")
        summary = f"Completed in {duration:.2f}s"
        if isinstance(val_loss, float):
            summary += f" | val/loss={val_loss:.6f}"
        print(summary)
        if args.print_metrics:
            print(row)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path)
    print(f"Wrote {len(frame)} benchmark rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
