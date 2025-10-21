#!/usr/bin/env python
"""CLI for running the live monitoring snapshot pipeline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from market_nn_plus_ultra.evaluation.operations import OperationsThresholds
from market_nn_plus_ultra.monitoring import DriftAlertThresholds, LiveMonitor, MonitoringSnapshot
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


@dataclass(slots=True)
class OperationsSnapshot:
    """Structured view of operations-summary artefacts."""

    risk: dict[str, float]
    guardrails: dict[str, float] | None
    alerts: list[str]


def _coerce_mapping(payload: dict[str, object] | None) -> dict[str, float] | None:
    if not payload:
        return None
    result: dict[str, float] = {}
    for key, value in payload.items():
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric):
            continue
        result[key] = numeric
    return result


def _load_operations_snapshot(path: Path) -> OperationsSnapshot:
    if not path.exists():
        raise FileNotFoundError(f"operations summary '{path}' does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "risk" not in payload:
        raise ValueError("operations summary is missing 'risk' metrics")
    risk = _coerce_mapping(payload["risk"]) or {}
    guardrails = _coerce_mapping(payload.get("guardrails"))
    alerts_raw = payload.get("triggered", [])
    alerts = [str(alert) for alert in alerts_raw if alert is not None]
    return OperationsSnapshot(risk=risk, guardrails=guardrails, alerts=alerts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference returns file (CSV or Parquet)")
    parser.add_argument(
        "live",
        type=Path,
        nargs="?",
        help="Live returns file (CSV or Parquet). Optional when --predictions/--evaluation-dir is provided.",
    )
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
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Evaluation predictions file (overrides the live positional argument)",
    )
    parser.add_argument(
        "--operations-summary",
        type=Path,
        help="Operations summary JSON produced by run_retraining_plan",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Directory containing evaluation artifacts from run_retraining_plan",
    )
    parser.add_argument(
        "--evaluation-predictions-name",
        type=str,
        default="predictions.parquet",
        help="Filename of the predictions artifact inside --evaluation-dir",
    )
    parser.add_argument(
        "--evaluation-operations-name",
        type=str,
        default="operations_summary.json",
        help="Filename of the operations summary artifact inside --evaluation-dir",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    reference = _load_returns(args.reference, args.return_col)

    live_path = args.predictions
    if live_path is None and args.live is not None:
        live_path = args.live
    if live_path is None and args.evaluation_dir is not None:
        live_path = args.evaluation_dir / args.evaluation_predictions_name
    if live_path is None:
        parser.error("a live returns file or --predictions/--evaluation-dir must be provided")
    live = _load_returns(live_path, args.return_col)

    operations_path = args.operations_summary
    if operations_path is None and args.evaluation_dir is not None:
        candidate = args.evaluation_dir / args.evaluation_operations_name
        if candidate.exists():
            operations_path = candidate

    operations_snapshot: OperationsSnapshot | None = None
    if operations_path is not None:
        operations_snapshot = _load_operations_snapshot(operations_path)

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

    if operations_snapshot is not None:
        drift_alerts = [
            alert
            for alert in snapshot.alerts
            if alert.startswith(
                (
                    "Population stability index",
                    "Jensen-Shannon divergence",
                    "KS statistic",
                )
            )
        ]
        combined_alerts = drift_alerts + operations_snapshot.alerts
        snapshot = MonitoringSnapshot(
            risk=operations_snapshot.risk,
            drift=snapshot.drift,
            alerts=combined_alerts,
            window_count=snapshot.window_count,
            guardrails=operations_snapshot.guardrails,
        )

    risk_table = format_metrics_table(snapshot.risk, precision=6)
    drift_table = format_metrics_table(snapshot.drift.as_dict(), precision=6)

    print("Risk metrics:")
    print(risk_table)
    print("\nDrift metrics:")
    print(drift_table)
    if snapshot.guardrails:
        guardrail_table = format_metrics_table(snapshot.guardrails, precision=6)
        print("\nGuardrail metrics:")
        print(guardrail_table)
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
