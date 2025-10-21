"""CLI entrypoint for running the Market NN Plus Ultra inference agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_nn_plus_ultra.training import load_experiment_from_file
from market_nn_plus_ultra.trading import MarketNNPlusUltraAgent
from market_nn_plus_ultra.utils import format_metrics_table, write_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Market NN Plus Ultra inference agent")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=Path, help="Optional Lightning checkpoint to restore")
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier (cpu, cuda:0, etc.)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agent_predictions.parquet"),
        help="Where to persist the predictions (parquet or csv)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip ROI metric computation even if realised returns are present",
    )
    parser.add_argument(
        "--return-column",
        type=str,
        default="realised_return",
        help="Column to treat as realised return when computing metrics",
    )
    parser.add_argument(
        "--benchmark-column",
        type=str,
        help="Optional benchmark return column for excess-return diagnostics",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional path to write evaluation metrics (json/csv/tsv/txt/md)",
    )
    parser.add_argument(
        "--metrics-format",
        type=str,
        choices=["json", "csv", "tsv", "txt", "md", "markdown"],
        help="Override the output format for metrics persistence",
    )
    return parser.parse_args()


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    config = load_experiment_from_file(args.config)
    agent = MarketNNPlusUltraAgent(
        experiment_config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    result = agent.run(
        evaluate=not args.no_eval,
        return_column=args.return_column,
        benchmark_column=args.benchmark_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(result.predictions, args.output)
    if result.metrics:
        print("\n=== Evaluation Metrics ===")
        print(format_metrics_table(result.metrics, precision=6))
        if args.metrics_output:
            fmt = args.metrics_format
            write_metrics_report(result.metrics, args.metrics_output, precision=6, format_hint=fmt)
            print(f"\nSaved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
