from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_nn_plus_ultra.evaluation import generate_report


def _load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        lines = path.suffix.lower() == ".jsonl"
        return pd.read_json(path, lines=lines)
    raise ValueError(f"Unsupported predictions format for '{path}'")


def _load_metrics(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("Metrics file must contain a JSON object mapping names to values")
    metrics: dict[str, float] = {}
    for key, value in data.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _load_milestones(path: Path | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as fp:
        data: Any = json.load(fp)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Milestones file must contain a list of objects or a single object")
    milestones: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        phase = entry.get("phase")
        milestone = entry.get("milestone") or entry.get("title") or entry.get("name")
        if not phase or not milestone:
            continue
        record: dict[str, str] = {"phase": str(phase), "milestone": str(milestone)}
        summary = entry.get("summary") or entry.get("notes")
        if summary is not None:
            record["summary"] = str(summary)
        milestones.append(record)
    return milestones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a performance report for Plus Ultra predictions")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions file (csv, parquet, json)")
    parser.add_argument("--output", type=Path, required=True, help="Destination report path (.md or .html)")
    parser.add_argument("--metrics", type=Path, help="Optional JSON file of additional metrics to include")
    parser.add_argument("--return-column", type=str, default="realised_return", help="Return column to evaluate")
    parser.add_argument(
        "--benchmark-column",
        type=str,
        help="Optional benchmark return column to compute excess metrics",
    )
    parser.add_argument("--title", type=str, default="Market NN Plus Ultra Performance Report", help="Report title")
    parser.add_argument("--description", type=str, help="Optional Markdown description to prefix the report")
    parser.add_argument("--periods-per-year", type=int, default=252, help="Return periods per year for annualisation")
    parser.add_argument("--format", choices=["auto", "markdown", "html"], default="auto", help="Force report format")
    parser.add_argument("--no-equity", action="store_true", help="Disable equity curve chart")
    parser.add_argument("--no-distribution", action="store_true", help="Disable return distribution chart")
    parser.add_argument(
        "--charts-dir-name",
        type=str,
        help="Override the directory name used to store generated charts",
    )
    parser.add_argument(
        "--milestones",
        type=Path,
        help="Optional JSON file containing research agenda milestone references",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = _load_predictions(args.predictions)
    metrics = _load_metrics(args.metrics)
    milestones = _load_milestones(args.milestones)

    output = args.output
    if args.format == "markdown" and output.suffix.lower() != ".md":
        output = output.with_suffix(".md")
    elif args.format == "html" and output.suffix.lower() != ".html":
        output = output.with_suffix(".html")

    generate_report(
        predictions,
        output,
        metrics=metrics,
        return_column=args.return_column,
        benchmark_column=args.benchmark_column,
        title=args.title,
        description=args.description,
        include_equity_chart=not args.no_equity,
        include_distribution_chart=not args.no_distribution,
        periods_per_year=args.periods_per_year,
        charts_dir_name=args.charts_dir_name,
        milestones=milestones,
    )


if __name__ == "__main__":
    main()

