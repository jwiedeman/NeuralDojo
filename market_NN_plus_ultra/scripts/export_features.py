"""CLI utility to export the feature registry into Markdown documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_nn_plus_ultra.data import FeatureRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Market NN Plus Ultra feature metadata")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated_features.md"),
        help="Destination Markdown file",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not embed a generation timestamp banner",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = FeatureRegistry()
    output_path = registry.to_markdown(
        args.output,
        include_timestamp=not args.no_timestamp,
    )
    print(f"Exported feature registry to {output_path}")


if __name__ == "__main__":
    main()

