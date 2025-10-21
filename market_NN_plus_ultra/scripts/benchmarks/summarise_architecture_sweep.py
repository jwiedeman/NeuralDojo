"""Summarise architecture sweep benchmarks into human-readable tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from market_nn_plus_ultra.evaluation.benchmarking import (
    architecture_leaderboard,
    dataframe_to_markdown,
    format_markdown_table,
    load_benchmark_frames,
    summarise_architecture_performance,
    summaries_to_frame,
)


def _parse_group_by(value: str | None) -> list[str]:
    if value is None:
        return []
    return [column.strip() for column in value.split(",") if column.strip()]


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
        "--leaderboard-group-by",
        type=str,
        help=(
            "Comma separated columns used to build leaderboards (e.g. "
            "'dataset_universe,dataset_split'). When omitted no leaderboard is generated."
        ),
    )
    parser.add_argument(
        "--leaderboard-top-k",
        type=int,
        default=3,
        help="Number of scenarios to keep per group when rendering the leaderboard (default: 3)",
    )
    parser.add_argument(
        "--leaderboard-format",
        choices=("markdown", "csv", "json", "parquet"),
        default="markdown",
        help="Output format for the optional leaderboard export (default: markdown)",
    )
    parser.add_argument(
        "--leaderboard-output",
        type=Path,
        help="Optional destination for the leaderboard export",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printing the formatted summary to stdout",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _write_output(path: Path, format_name: str, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "markdown":
        path.write_text(content, encoding="utf-8")
    elif format_name == "csv":
        content.to_csv(path, index=False)
    elif format_name == "json":
        content.to_json(path, orient="records", indent=2)
    elif format_name == "parquet":
        content.to_parquet(path, index=False)
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

    leaderboard_group_by = _parse_group_by(args.leaderboard_group_by)
    if leaderboard_group_by:
        leaderboard = architecture_leaderboard(
            frame,
            metric=args.metric,
            higher_is_better=args.higher_is_better,
            group_by=leaderboard_group_by,
            top_k=args.leaderboard_top_k,
        )
        if args.leaderboard_format == "markdown":
            leaderboard_text = dataframe_to_markdown(leaderboard)
            if not args.quiet:
                print()
                print("Leaderboard:")
                print(leaderboard_text)
            if args.leaderboard_output:
                _write_output(args.leaderboard_output, "markdown", leaderboard_text)
        else:
            if args.leaderboard_format == "csv":
                rendered = leaderboard.to_csv(index=False)
            elif args.leaderboard_format == "json":
                rendered = leaderboard.to_json(orient="records", indent=2)
            else:  # parquet
                rendered = leaderboard
            if not args.quiet:
                print()
                if args.leaderboard_format == "parquet":
                    print("Leaderboard (parquet preview):")
                    print(leaderboard.to_string(index=False))
                else:
                    print("Leaderboard:")
                    print(rendered)
            if args.leaderboard_output:
                if args.leaderboard_format == "parquet":
                    _write_output(args.leaderboard_output, "parquet", leaderboard)
                elif args.leaderboard_format == "csv":
                    _write_output(args.leaderboard_output, "csv", leaderboard)
                elif args.leaderboard_format == "json":
                    _write_output(args.leaderboard_output, "json", leaderboard)

    return 0


if __name__ == "__main__":
    sys.exit(main())
