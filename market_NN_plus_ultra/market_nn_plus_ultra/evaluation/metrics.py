"""Risk-adjusted evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_aggregate_metrics(equity_curve: pd.Series) -> dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    mean_return = returns.mean()
    volatility = returns.std()
    sharpe = mean_return / (volatility + 1e-9) * np.sqrt(252)
    downside = returns[returns < 0].std()
    sortino = mean_return / (downside + 1e-9) * np.sqrt(252)
    rolling_max = equity_curve.cummax()
    drawdowns = equity_curve / (rolling_max + 1e-9) - 1
    max_drawdown = drawdowns.min()
    calmar = -mean_return / (max_drawdown + 1e-9)
    return {
        "annual_return": (1 + mean_return) ** 252 - 1,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def evaluate_policy(trades: pd.DataFrame) -> dict[str, float]:
    """Compute evaluation metrics from executed trades.

    Args:
        trades: DataFrame with ``timestamp``, ``pnl`` and ``equity`` columns.
    """

    equity_curve = trades.set_index("timestamp")["equity"].sort_index()
    metrics = compute_aggregate_metrics(equity_curve)
    metrics["hit_rate"] = (trades["pnl"] > 0).mean()
    metrics["avg_trade"] = trades["pnl"].mean()
    metrics["total_return"] = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    return metrics
