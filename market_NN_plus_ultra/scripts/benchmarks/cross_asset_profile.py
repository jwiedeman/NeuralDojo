"""Profile cross-asset alignment statistics for a Plus Ultra SQLite dataset."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from market_nn_plus_ultra.data.cross_asset import build_cross_asset_view
from market_nn_plus_ultra.data.validation import validate_cross_asset_view_frame


def _parse_columns(raw: str | None, *, default: Sequence[str]) -> list[str]:
    if raw is None:
        return list(default)
    items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return items or list(default)


def _parse_symbols(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    symbols = {chunk.strip() for chunk in raw.split(",") if chunk.strip()}
    return symbols or None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path, help="Path to the SQLite database containing a series table")
    parser.add_argument(
        "--columns",
        type=str,
        help="Comma separated list of series columns to align (default: close,volume)",
    )
    parser.add_argument(
        "--fill-limit",
        type=int,
        help="Maximum consecutive rows to forward/back-fill when aligning (default: unlimited)",
    )
    parser.add_argument(
        "--no-returns",
        action="store_true",
        help="Disable automatic log-return feature generation",
    )
    parser.add_argument(
        "--symbol-universe",
        type=str,
        help="Optional comma separated subset of symbols to profile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination parquet path for the generated cross-asset view",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    db_path = args.db_path.resolve()
    if not db_path.exists():
        print(f"error: database '{db_path}' does not exist", file=sys.stderr)
        return 1

    with sqlite3.connect(db_path) as conn:
        series = pd.read_sql_query("SELECT * FROM series", conn, parse_dates=["timestamp"])

    if series.empty:
        print("error: series table is empty", file=sys.stderr)
        return 1

    symbols = _parse_symbols(args.symbol_universe)
    if symbols is not None:
        series = series[series["symbol"].isin(symbols)]
        if series.empty:
            print("error: no rows remain after applying symbol filter", file=sys.stderr)
            return 1

    columns = _parse_columns(args.columns, default=("close", "volume"))

    start = time.perf_counter()
    result = build_cross_asset_view(
        series,
        value_columns=columns,
        include_returns=not args.no_returns,
        fill_limit=args.fill_limit,
    )
    duration = time.perf_counter() - start
    view = validate_cross_asset_view_frame(result.frame)

    stats = result.stats.to_dict()
    stats.update(
        {
            "rows": int(len(view)),
            "duration_seconds": duration,
            "columns": columns,
            "include_returns": bool(not args.no_returns),
        }
    )
    print(json.dumps(stats, indent=2, sort_keys=True))

    if args.output is not None:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        view.to_parquet(destination)
        print(f"wrote cross-asset view to {destination}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
