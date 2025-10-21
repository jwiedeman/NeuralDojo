#!/usr/bin/env python
"""CLI for running the live monitoring snapshot pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from market_nn_plus_ultra.evaluation.operations import OperationsThresholds
from market_nn_plus_ultra.monitoring import (
    DriftAlertThresholds,
    LiveMonitor,
)
from market_nn_plus_ultra.utils.reporting import format_metrics_table


def _load_returns(path: Path, column: str) -> Iterable[float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"returns file '{path}' does not exist")
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported file format for '{path}'")
    if column not in frame:
        raise ValueError(f"column '{column}' missing from '{path}'")
    series = frame[column].astype(float).dropna()
    return series.to_numpy(dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference returns file (CSV or Parquet)")
    parser.add_argument("live", type=Path, help="Live returns file (CSV or Parquet)")
    parser.add_argument("--return-col", default="realised_return", help="Column containing returns")
    parser.add_argument("--window", type=int, default=512, help="Rolling window size for monitoring")
    parser.add_argument("--drift-bins", type=int, default=20, help="Number of bins for drift histograms")
    parser.add_argument("--min-sharpe", type=float, default=None, help="Minimum Sharpe ratio threshold")
    parser.add_argument("--max-drawdown", type=float, default=None, help="Maximum drawdown magnitude threshold")
    parser.add_argument("--min-tail-return", type=float, default=None, help="Minimum acceptable Value-at-Risk")
    parser.add_argument("--psi-alert", type=float, default=None, help="PSI alert threshold")
    parser.add_argument("--js-alert", type=float, default=None, help="JS divergence alert threshold")
    parser.add_argument("--ks-alert", type=float, default=None, help="KS statistic alert threshold")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    reference = _load_returns(args.reference, args.return_col)
    live = _load_returns(args.live, args.return_col)

    risk_thresholds = OperationsThresholds(
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        min_tail_return=args.min_tail_return,
    )
    default_drift = DriftAlertThresholds()
    drift_thresholds = DriftAlertThresholds(
        psi_alert=args.psi_alert if args.psi_alert is not None else default_drift.psi_alert,
        js_alert=args.js_alert if args.js_alert is not None else default_drift.js_alert,
        ks_alert=args.ks_alert if args.ks_alert is not None else default_drift.ks_alert,
    )

    monitor = LiveMonitor(
        reference,
        window_size=args.window,
        drift_bins=args.drift_bins,
        risk_thresholds=risk_thresholds,
        drift_thresholds=drift_thresholds,
    )

    snapshot = monitor.update(live)

    risk_table = format_metrics_table(snapshot.risk, precision=6)
    drift_table = format_metrics_table(snapshot.drift.as_dict(), precision=6)

    print("Risk metrics:")
    print(risk_table)
    print("\nDrift metrics:")
    print(drift_table)
    if snapshot.alerts:
        print("\nAlerts:")
        for alert in snapshot.alerts:
            print(f"- {alert}")
    else:
        print("\nAlerts: none")

    if args.output is not None:
        payload = snapshot.as_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
