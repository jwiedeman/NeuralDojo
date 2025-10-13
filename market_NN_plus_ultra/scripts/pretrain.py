"""CLI entrypoint for running masked time-series pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_nn_plus_ultra.training import load_experiment_from_file, run_pretraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run masked time-series pretraining for Market NN Plus Ultra")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_from_file(args.config)
    if config.pretraining is None:
        raise SystemExit("Config must include a 'pretraining' section for self-supervised runs")
    result = run_pretraining(config)
    print(f"Best checkpoint stored at: {result['best_model_path']}")


if __name__ == "__main__":
    main()
