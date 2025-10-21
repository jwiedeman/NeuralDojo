from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from market_nn_plus_ultra.evaluation.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
)


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported file format for '{path}'")


def _write_metrics(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
        return
    if suffix in {".csv", ".txt"}:
        frame.to_csv(destination, index=False)
        return
    if suffix == ".tsv":
        frame.to_csv(destination, index=False, sep="\t")
        return
    raise ValueError(f"Unsupported metrics output format for '{destination}'")


def _normalise_indent(indent: int | None) -> int | None:
    if indent is None or indent <= 0:
        return None
    return indent


def _timestamp_to_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return str(value)


def _summarise_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for column in frame.columns:
        if not column.startswith("metric_"):
            continue
        values = frame[column].dropna().astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            stats = {"mean": None, "median": None, "std": None, "min": None, "max": None}
        else:
            std = float(finite.std(ddof=0)) if finite.size > 1 else 0.0
            stats = {
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "std": std,
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
        summary[column.removeprefix("metric_")] = stats
    return summary


def _split_descriptor(row: pd.Series) -> dict[str, str | None]:
    return {
        "train_start": _timestamp_to_str(row.get("train_start")),
        "train_end": _timestamp_to_str(row.get("train_end")),
        "test_start": _timestamp_to_str(row.get("test_start")),
        "test_end": _timestamp_to_str(row.get("test_end")),
    }


def _describe_extremes(frame: pd.DataFrame) -> dict[str, dict[str, str | None]]:
    summary: dict[str, dict[str, str | None]] = {}
    if frame.empty:
        return summary
    if "metric_sharpe" in frame.columns:
        best_idx = frame["metric_sharpe"].astype(float).idxmax()
        if pd.notna(best_idx):
            best_row = frame.loc[best_idx]
            summary["best_sharpe_split"] = _split_descriptor(best_row)
    if "metric_max_drawdown" in frame.columns:
        worst_idx = frame["metric_max_drawdown"].astype(float).idxmin()
        if pd.notna(worst_idx):
            worst_row = frame.loc[worst_idx]
            summary["worst_drawdown_split"] = _split_descriptor(worst_row)
    return summary


def _build_summary(frame: pd.DataFrame, config: WalkForwardConfig) -> dict[str, object]:
    summary: dict[str, object] = {
        "splits": int(len(frame)),
        "train_window": config.train_window,
        "test_window": config.test_window,
        "step": config.step or config.test_window,
        "timestamp_column": config.timestamp_column,
        "return_column": config.return_column,
        "metrics": _summarise_metrics(frame),
    }
    summary.update(_describe_extremes(frame))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward evaluation over realised returns and emit metrics",
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Path to the predictions dataset")
    parser.add_argument(
        "--train-window",
        type=int,
        required=True,
        help="Number of periods in each training window",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        required=True,
        help="Number of periods in each evaluation window",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Advance the window by this many periods (defaults to test window)",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="window_end",
        help="Timestamp column in the predictions dataset",
    )
    parser.add_argument(
        "--return-column",
        type=str,
        default="realised_return",
        help="Realised return column in the predictions dataset",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=1,
        help="Minimum number of rows required for each training window",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("benchmarks/walkforward_metrics.parquet"),
        help="Destination for the per-split metrics table",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional JSON file capturing aggregated statistics",
    )
    parser.add_argument(
        "--summary-indent",
        type=int,
        default=2,
        help="Number of spaces used to pretty-print the JSON summary",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary printing; useful for scripting",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    predictions = _load_frame(args.predictions)
    config = WalkForwardConfig(
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        timestamp_column=args.timestamp_column,
        return_column=args.return_column,
        min_train_size=args.min_train_size,
    )
    backtester = WalkForwardBacktester(config)
    metrics = backtester.run(predictions)
    _write_metrics(metrics, args.metrics_output)
    summary = _build_summary(metrics, config)
    return metrics, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _, summary = run(args)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=_normalise_indent(args.summary_indent), sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        print(json.dumps(summary, indent=_normalise_indent(args.summary_indent), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

