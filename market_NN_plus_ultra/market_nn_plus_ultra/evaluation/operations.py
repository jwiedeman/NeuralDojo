"""Utilities for producing operations-ready summaries of model runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import guardrail_metrics, risk_metrics


@dataclass(slots=True)
class OperationsThresholds:
    """Thresholds that trigger guardrail alerts in the summary output."""

    min_sharpe: float | None = None
    max_drawdown: float | None = None
    max_gross_exposure: float | None = None
    max_turnover: float | None = None
    min_tail_return: float | None = None
    max_tail_frequency: float | None = None
    max_symbol_exposure: float | None = None


@dataclass(slots=True)
class OperationsSummary:
    """Aggregate snapshot of risk metrics, guardrails, and triggered alerts."""

    risk: Dict[str, float]
    guardrails: Dict[str, float] | None
    triggered: List[str]

    def as_dict(self) -> Dict[str, object]:
        """Return the summary payload as a serialisable dictionary."""

        payload: Dict[str, object] = {
            "risk": dict(self.risk),
            "triggered": list(self.triggered),
        }
        if self.guardrails is not None:
            payload["guardrails"] = dict(self.guardrails)
        return payload


def compile_operations_summary(
    predictions: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    return_col: str = "realised_return",
    benchmark_col: str | None = None,
    trade_timestamp_col: str = "timestamp",
    trade_symbol_col: str = "symbol",
    trade_notional_col: str = "notional",
    trade_position_col: str = "position",
    trade_price_col: str = "price",
    trade_return_col: str = "pnl",
    capital_base: float = 1.0,
    tail_percentile: float = 5.0,
    thresholds: OperationsThresholds | None = None,
) -> OperationsSummary:
    """Compile a production readiness summary for a run.

    Parameters
    ----------
    predictions:
        Frame containing realised returns for the evaluation window.
    trades:
        Optional trade log for exposure and turnover guardrails.
    return_col:
        Column in ``predictions`` holding realised returns.
    trade_timestamp_col, trade_symbol_col, trade_notional_col, trade_position_col,
    trade_price_col, trade_return_col:
        Column names expected in ``trades`` when guardrail diagnostics are
        computed.
    capital_base:
        Reference capital used to normalise guardrail metrics.
    tail_percentile:
        Percentile used for tail-return calculations.
    thresholds:
        Optional threshold configuration. When supplied, metrics breaching a
        threshold produce human-readable alerts in ``OperationsSummary.triggered``.
    benchmark_col:
        Optional realised-return column containing benchmark performance. When
        provided, the summary includes excess-return, tracking-error, beta, and
        information-ratio diagnostics relative to the benchmark.
    """

    if return_col not in predictions:
        raise ValueError(f"return column '{return_col}' is missing from predictions")

    returns_series = pd.to_numeric(predictions[return_col], errors="coerce")
    benchmark_array: np.ndarray | None = None
    if benchmark_col is not None:
        if benchmark_col not in predictions:
            raise ValueError(
                f"benchmark column '{benchmark_col}' is missing from predictions"
            )
        benchmark_series = pd.to_numeric(predictions[benchmark_col], errors="coerce")
        aligned = pd.concat(
            [
                returns_series.rename("__returns"),
                benchmark_series.rename("__benchmark"),
            ],
            axis=1,
        ).dropna()
        returns_array = aligned["__returns"].to_numpy(dtype=np.float64)
        benchmark_array = aligned["__benchmark"].to_numpy(dtype=np.float64)
    else:
        returns_array = returns_series.dropna().to_numpy(dtype=np.float64)

    risk = risk_metrics(returns_array, benchmark_returns=benchmark_array)

    guardrails: Dict[str, float] | None = None
    if trades is not None:
        if not isinstance(trades, pd.DataFrame):
            raise TypeError("trades must be a pandas DataFrame when provided")
        if not trades.empty:
            guardrails = guardrail_metrics(
                trades,
                timestamp_col=trade_timestamp_col,
                symbol_col=trade_symbol_col,
                notional_col=trade_notional_col,
                position_col=trade_position_col,
                price_col=trade_price_col,
                return_col=trade_return_col,
                capital_base=capital_base,
                tail_percentile=tail_percentile,
            )

    triggered: List[str] = []
    cfg = thresholds or OperationsThresholds()

    sharpe = float(risk.get("sharpe", 0.0))
    if cfg.min_sharpe is not None and sharpe < cfg.min_sharpe:
        triggered.append(
            f"Sharpe ratio {sharpe:.3f} fell below minimum target {cfg.min_sharpe:.3f}"
        )

    drawdown_value = float(abs(risk.get("max_drawdown", 0.0)))
    if cfg.max_drawdown is not None and drawdown_value > cfg.max_drawdown:
        triggered.append(
            f"Max drawdown {drawdown_value:.3f} exceeded limit {cfg.max_drawdown:.3f}"
        )

    if guardrails:
        gross_peak = float(guardrails.get("gross_exposure_peak", 0.0))
        if cfg.max_gross_exposure is not None and gross_peak > cfg.max_gross_exposure:
            triggered.append(
                "Gross exposure peak "
                f"{gross_peak:.3f} exceeded limit {cfg.max_gross_exposure:.3f}"
            )

        turnover = float(guardrails.get("turnover_rate", 0.0))
        if cfg.max_turnover is not None and turnover > cfg.max_turnover:
            triggered.append(
                f"Turnover rate {turnover:.3f} exceeded limit {cfg.max_turnover:.3f}"
            )

        tail_quantile = float(guardrails.get("tail_return_quantile", 0.0))
        if cfg.min_tail_return is not None and tail_quantile < cfg.min_tail_return:
            triggered.append(
                "Tail return quantile "
                f"{tail_quantile:.3f} breached floor {cfg.min_tail_return:.3f}"
            )

        tail_frequency = float(guardrails.get("tail_event_frequency", 0.0))
        if cfg.max_tail_frequency is not None and tail_frequency > cfg.max_tail_frequency:
            triggered.append(
                "Tail event frequency "
                f"{tail_frequency:.3f} exceeded limit {cfg.max_tail_frequency:.3f}"
            )

        symbol_peak = float(guardrails.get("max_symbol_exposure", 0.0))
        if cfg.max_symbol_exposure is not None and symbol_peak > cfg.max_symbol_exposure:
            triggered.append(
                "Symbol exposure peak "
                f"{symbol_peak:.3f} exceeded limit {cfg.max_symbol_exposure:.3f}"
            )

    return OperationsSummary(risk=risk, guardrails=guardrails, triggered=triggered)


__all__ = [
    "OperationsSummary",
    "OperationsThresholds",
    "compile_operations_summary",
]

