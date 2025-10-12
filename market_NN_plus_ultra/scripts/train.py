"""CLI entrypoint for training Market NN Plus Ultra models."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_nn_plus_ultra.training import load_experiment_from_file, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Market NN Plus Ultra trading agent")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Experiment YAML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_from_file(args.config)
    run_training(config)


if __name__ == "__main__":
    main()

