"""CLI entrypoint for running the Market NN Plus Ultra inference agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_nn_plus_ultra.training import load_experiment_from_file
from market_nn_plus_ultra.trading import MarketNNPlusUltraAgent


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
    result = agent.run(evaluate=not args.no_eval, return_column=args.return_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(result.predictions, args.output)
    if result.metrics:
        for name, value in result.metrics.items():
            print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
