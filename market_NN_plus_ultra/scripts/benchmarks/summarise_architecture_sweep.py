"""Summarise architecture sweep outputs into digestible comparison tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from market_nn_plus_ultra.evaluation.benchmarking import (
    format_markdown_table,
    load_benchmark_frames,
    summarise_architecture_performance,
    summaries_to_frame,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more benchmark parquet/CSV/JSON files produced by the architecture sweep",
    )
    parser.add_argument(
        "--metric",
        default="metric_val_loss",
        help="Metric column used to rank architectures (default: metric_val_loss)",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger metric values as better (useful for ROI-style metrics)",
    )
    parser.add_argument(
        "--profitability-metric",
        default="profitability_roi",
        help="Optional profitability metric to average per architecture",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json", "parquet"),
        default="markdown",
        help="Output format when writing to --output (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination file for the summary (extension does not determine format)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printing the formatted summary to stdout",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _write_output(path: Path, format_name: str, summaries_frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "markdown":
        path.write_text(summaries_frame, encoding="utf-8")
    elif format_name == "csv":
        summaries_frame.to_csv(path, index=False)
    elif format_name == "json":
        summaries_frame.to_json(path, orient="records", indent=2)
    elif format_name == "parquet":
        summaries_frame.to_parquet(path, index=False)
    else:  # pragma: no cover - argparse guards choices
        raise ValueError(f"Unsupported format: {format_name}")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    frame = load_benchmark_frames(args.inputs)
    summaries = summarise_architecture_performance(
        frame,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
        profitability_metric=args.profitability_metric,
    )

    if args.format == "markdown":
        formatted = format_markdown_table(summaries, metric=args.metric)
        if not args.quiet:
            print(formatted)
        if args.output:
            _write_output(args.output, "markdown", formatted)
    else:
        summary_frame = summaries_to_frame(summaries)
        if not args.quiet:
            if args.format == "csv":
                print(summary_frame.to_csv(index=False))
            elif args.format == "json":
                print(summary_frame.to_json(orient="records", indent=2))
            else:  # parquet
                print(summary_frame.to_string(index=False))
        if args.output:
            _write_output(args.output, args.format, summary_frame)

    return 0


if __name__ == "__main__":
    sys.exit(main())
