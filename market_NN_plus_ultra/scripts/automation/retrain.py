"""CLI entry point for the continuous retraining orchestrator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from market_nn_plus_ultra.automation import (
    DatasetStageConfig,
    EvaluationStageConfig,
    RetrainingPlan,
    WarmStartStrategy,
    run_retraining_plan,
)
from market_nn_plus_ultra.evaluation import OperationsThresholds


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
    parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help="Run inference and operations monitoring after training",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        help="Optional experiment YAML dedicated to evaluation (defaults to --train-config)",
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        help="Override the checkpoint used during evaluation",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        help="Directory for evaluation artifacts (defaults to <output-dir>/evaluation)",
    )
    parser.add_argument(
        "--eval-predictions",
        type=Path,
        help="Where to write evaluation predictions (relative paths resolved under the evaluation output dir)",
    )
    parser.add_argument(
        "--eval-metrics",
        type=Path,
        help="Where to write evaluation metrics JSON",
    )
    parser.add_argument(
        "--eval-operations",
        type=Path,
        help="Where to write the operations summary JSON",
    )
    parser.add_argument(
        "--eval-trades",
        type=Path,
        help="Optional trades log used for guardrail diagnostics",
    )
    parser.add_argument(
        "--eval-return-col",
        type=str,
        default="realised_return",
        help="Realised return column expected in the predictions dataset",
    )
    parser.add_argument(
        "--eval-benchmark-col",
        type=str,
        help="Optional benchmark return column for excess diagnostics",
    )
    parser.add_argument(
        "--eval-device",
        type=str,
        default="cpu",
        help="Device used for evaluation inference (cpu, cuda:0, etc.)",
    )
    parser.add_argument(
        "--eval-trade-timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column expected in the trades log",
    )
    parser.add_argument(
        "--eval-trade-symbol-col",
        type=str,
        default="symbol",
        help="Symbol column expected in the trades log",
    )
    parser.add_argument(
        "--eval-trade-notional-col",
        type=str,
        default="notional",
        help="Notional exposure column expected in the trades log",
    )
    parser.add_argument(
        "--eval-trade-position-col",
        type=str,
        default="position",
        help="Position column expected in the trades log",
    )
    parser.add_argument(
        "--eval-trade-price-col",
        type=str,
        default="price",
        help="Price column expected in the trades log",
    )
    parser.add_argument(
        "--eval-trade-return-col",
        type=str,
        default="pnl",
        help="Return/PnL column expected in the trades log",
    )
    parser.add_argument(
        "--eval-capital-base",
        type=float,
        default=1.0,
        help="Reference capital used for guardrail normalisation",
    )
    parser.add_argument(
        "--eval-tail-percentile",
        type=float,
        default=5.0,
        help="Percentile used for tail-return guardrails",
    )
    parser.add_argument("--eval-min-sharpe", type=float, help="Minimum Sharpe ratio before raising an alert")
    parser.add_argument("--eval-max-drawdown", type=float, help="Maximum drawdown magnitude allowed")
    parser.add_argument(
        "--eval-max-gross-exposure",
        type=float,
        help="Maximum gross exposure relative to capital",
    )
    parser.add_argument("--eval-max-turnover", type=float, help="Maximum turnover rate allowed")
    parser.add_argument(
        "--eval-min-tail-return",
        type=float,
        help="Minimum acceptable tail-return quantile",
    )
    parser.add_argument(
        "--eval-max-tail-frequency",
        type=float,
        help="Maximum tail-event frequency allowed",
    )
    parser.add_argument(
        "--eval-max-symbol-exposure",
        type=float,
        help="Maximum symbol exposure relative to capital",
    )
    return parser


def _resolve_output_dir(base: Path | None) -> Path:
    if base is not None:
        return base
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("automation_runs") / timestamp


def _build_evaluation_stage(args: argparse.Namespace) -> EvaluationStageConfig:
    thresholds = OperationsThresholds(
        min_sharpe=args.eval_min_sharpe,
        max_drawdown=args.eval_max_drawdown,
        max_gross_exposure=args.eval_max_gross_exposure,
        max_turnover=args.eval_max_turnover,
        min_tail_return=args.eval_min_tail_return,
        max_tail_frequency=args.eval_max_tail_frequency,
        max_symbol_exposure=args.eval_max_symbol_exposure,
    )
    return EvaluationStageConfig(
        output_dir=args.eval_output,
        predictions_path=args.eval_predictions,
        metrics_path=args.eval_metrics,
        operations_path=args.eval_operations,
        trades_path=args.eval_trades,
        return_column=args.eval_return_col,
        benchmark_column=args.eval_benchmark_col,
        device=args.eval_device,
        trade_timestamp_col=args.eval_trade_timestamp_col,
        trade_symbol_col=args.eval_trade_symbol_col,
        trade_notional_col=args.eval_trade_notional_col,
        trade_position_col=args.eval_trade_position_col,
        trade_price_col=args.eval_trade_price_col,
        trade_return_col=args.eval_trade_return_col,
        capital_base=args.eval_capital_base,
        tail_percentile=args.eval_tail_percentile,
        operations_thresholds=thresholds,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_dir = _resolve_output_dir(args.output_dir)

    dataset_stage = DatasetStageConfig(
        strict_validation=not args.no_validation,
        regenerate_regimes=args.regenerate_regimes,
    )

    evaluation_stage = _build_evaluation_stage(args)

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
        evaluation_config=args.eval_config,
        run_evaluation=args.run_evaluation,
        evaluation_checkpoint=args.eval_checkpoint,
        evaluation_stage=evaluation_stage,
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

