"""CLI entry-point for training Market NN Plus Ultra models."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

def _load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Market NN Plus Ultra trader")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    # TODO: parse into dataclasses and invoke Trainer once implementation is complete.
    print("Loaded configuration:")
    print(cfg)


if __name__ == "__main__":
    main()
