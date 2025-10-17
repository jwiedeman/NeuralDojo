#!/usr/bin/env python
"""CLI wrapper around the synthetic fixture generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_nn_plus_ultra.data.fixtures import FixtureConfig, build_fixture, write_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SQLite fixture for Market NN Plus Ultra experiments.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the SQLite database that will be created.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ALPHA", "BETA", "GAMMA", "DELTA"],
        help="Symbols to include in the fixture (default: %(default)s).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=32_768,
        help="Number of rows per symbol to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--freq",
        default="15min",
        help="Pandas offset alias controlling the candle frequency (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible generation (default: %(default)s).",
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="ISO date to start the generated timeline (default: %(default)s).",
    )
    parser.add_argument(
        "--alt-features",
        type=int,
        default=3,
        help="Number of synthetic alternative data signals per symbol (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FixtureConfig(
        symbols=list(dict.fromkeys(args.symbols)),
        rows=args.rows,
        freq=args.freq,
        seed=args.seed,
        start=pd.Timestamp(args.start, tz=None).to_pydatetime(),
        alt_features=max(0, args.alt_features),
    )
    frames = build_fixture(config)
    write_fixture(frames, args.output)
    print(
        f"Wrote fixture with {len(config.symbols)} symbols, {config.rows} rows each, "
        f"frequency {config.freq} to '{args.output}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
