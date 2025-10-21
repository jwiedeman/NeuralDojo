"""CLI for compiling operations readiness summaries.

This script wraps :func:`market_nn_plus_ultra.evaluation.compile_operations_summary`
so analysts can turn prediction and trade logs into structured JSON payloads.
It mirrors the thresholds exposed by :class:`OperationsThresholds` and aims to
keep automation/reporting pipelines in sync with the optimisation plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from market_nn_plus_ultra.evaluation import (
    OperationsThresholds,
    compile_operations_summary,
)


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported file format for '{path}'")


def _maybe_load(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Could not locate '{path}'")
    return _load_frame(path)


def _parse_thresholds(args: argparse.Namespace) -> OperationsThresholds:
    return OperationsThresholds(
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        max_gross_exposure=args.max_gross_exposure,
        max_turnover=args.max_turnover,
        min_tail_return=args.min_tail_return,
        max_tail_frequency=args.max_tail_frequency,
        max_symbol_exposure=args.max_symbol_exposure,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile an operations readiness summary for a Plus Ultra run",
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Path to the predictions dataset")
    parser.add_argument("--trades", type=Path, help="Optional path to the trade log for guardrail metrics")
    parser.add_argument("--output", type=Path, help="Optional JSON destination; prints to stdout when omitted")
    parser.add_argument("--return-column", type=str, default="realised_return", help="Realised return column in predictions")
    parser.add_argument("--trade-timestamp-col", type=str, default="timestamp", help="Timestamp column in the trade log")
    parser.add_argument("--trade-symbol-col", type=str, default="symbol", help="Symbol column in the trade log")
    parser.add_argument("--trade-notional-col", type=str, default="notional", help="Notional exposure column in the trade log")
    parser.add_argument("--trade-position-col", type=str, default="position", help="Position column in the trade log")
    parser.add_argument("--trade-price-col", type=str, default="price", help="Price column in the trade log")
    parser.add_argument("--trade-return-col", type=str, default="pnl", help="Return or PnL column in the trade log")
    parser.add_argument(
        "--capital-base",
        type=float,
        default=1.0,
        help="Reference capital used to normalise guardrail diagnostics",
    )
    parser.add_argument(
        "--tail-percentile",
        type=float,
        default=5.0,
        help="Percentile used to compute tail-return guardrails",
    )
    parser.add_argument("--min-sharpe", type=float, help="Minimum Sharpe ratio before flagging an alert")
    parser.add_argument("--max-drawdown", type=float, help="Maximum allowable drawdown magnitude")
    parser.add_argument("--max-gross-exposure", type=float, help="Maximum gross exposure peak relative to capital")
    parser.add_argument("--max-turnover", type=float, help="Maximum turnover rate allowed")
    parser.add_argument("--min-tail-return", type=float, help="Minimum acceptable tail-return quantile")
    parser.add_argument("--max-tail-frequency", type=float, help="Maximum tail-event frequency allowed")
    parser.add_argument("--max-symbol-exposure", type=float, help="Maximum symbol exposure relative to capital")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Number of spaces to indent JSON output (use 0 for compact output)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def _normalise_indent(indent: int | None) -> int | None:
    if indent is None:
        return None
    if indent <= 0:
        return None
    return indent


def _dump(payload: dict[str, Any], output: Path | None, indent: int | None) -> None:
    json_payload = json.dumps(payload, indent=_normalise_indent(indent), sort_keys=True)
    if output is None:
        print(json_payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_payload + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictions = _load_frame(args.predictions)
    trades = _maybe_load(args.trades)
    summary = compile_operations_summary(
        predictions,
        trades,
        return_col=args.return_column,
        trade_timestamp_col=args.trade_timestamp_col,
        trade_symbol_col=args.trade_symbol_col,
        trade_notional_col=args.trade_notional_col,
        trade_position_col=args.trade_position_col,
        trade_price_col=args.trade_price_col,
        trade_return_col=args.trade_return_col,
        capital_base=args.capital_base,
        tail_percentile=args.tail_percentile,
        thresholds=_parse_thresholds(args),
    )
    payload = summary.as_dict()
    payload["triggered"] = summary.triggered
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    _dump(payload, args.output, args.indent)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

