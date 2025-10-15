#!/usr/bin/env python
"""Compile the Market NN Plus Ultra package to bytecode as a fast syntax smoke test."""

from __future__ import annotations

import argparse
import compileall
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the market_nn_plus_ultra package to verify syntax integrity.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "market_nn_plus_ultra",
        type=Path,
        help="Path to the package directory (defaults to the project package).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation even if timestamps appear up to date.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes to use (0 lets compileall pick an optimal value).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum directory depth to recurse when compiling modules.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file compilation output (only errors are shown).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_path = args.path.resolve()
    if not package_path.exists():
        print(f"error: path '{package_path}' does not exist", file=sys.stderr)
        return 2

    quiet = 2 if args.quiet else 0
    success = compileall.compile_dir(
        dir=str(package_path),
        maxlevels=args.max_depth,
        force=args.force,
        quiet=quiet,
        workers=args.workers,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
