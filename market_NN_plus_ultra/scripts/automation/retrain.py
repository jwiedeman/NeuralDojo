"""CLI entry point for the continuous retraining orchestrator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from market_nn_plus_ultra.automation import (
    DatasetStageConfig,
    RetrainingPlan,
    WarmStartStrategy,
    run_retraining_plan,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dataset validation, training, and PPO fine-tuning as a single workflow.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the SQLite database")
    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Experiment YAML used for supervised training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where orchestration artifacts should be stored (defaults to ./automation_runs/<timestamp>)",
    )
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        help="Optional experiment YAML for self-supervised pretraining",
    )
    parser.add_argument(
        "--reinforcement-config",
        type=Path,
        help="Optional experiment YAML dedicated to PPO runs (defaults to --train-config)",
    )
    parser.add_argument(
        "--skip-pretraining",
        action="store_true",
        help="Skip the pretraining stage even if --pretrain-config is provided",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the supervised training stage",
    )
    parser.add_argument(
        "--run-reinforcement",
        action="store_true",
        help="Enable PPO fine-tuning after supervised training",
    )
    parser.add_argument(
        "--warm-start",
        choices=["training", "pretraining", "none"],
        default="training",
        help="Checkpoint source for PPO fine-tuning",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip loading the entire dataset during validation (schema checks still run)",
    )
    parser.add_argument(
        "--regenerate-regimes",
        action="store_true",
        help="Regenerate regime labels before training",
    )
    return parser


def _resolve_output_dir(base: Path | None) -> Path:
    if base is not None:
        return base
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("automation_runs") / timestamp


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_dir = _resolve_output_dir(args.output_dir)

    dataset_stage = DatasetStageConfig(
        strict_validation=not args.no_validation,
        regenerate_regimes=args.regenerate_regimes,
    )

    plan = RetrainingPlan(
        dataset_path=args.dataset,
        training_config=args.train_config,
        output_dir=output_dir,
        dataset_stage=dataset_stage,
        pretraining_config=args.pretrain_config,
        run_pretraining=not args.skip_pretraining and args.pretrain_config is not None,
        run_training=not args.skip_training,
        reinforcement_config=args.reinforcement_config,
        run_reinforcement=args.run_reinforcement,
        warm_start=WarmStartStrategy.from_arg(args.warm_start),
    )

    summary = run_retraining_plan(plan)
    print(f"Orchestration complete — stages executed: {[stage.name for stage in summary.stages]}")
    for stage in summary.stages:
        print(f"[{stage.name}] success={stage.success} duration={stage.duration_seconds:.2f}s")
        if stage.artifacts:
            print(f"    artifacts: {stage.artifacts}")
        for note in stage.notes:
            print(f"    note: {note}")


if __name__ == "__main__":  # pragma: no cover - exercised via CLI tests
    main()

