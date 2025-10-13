"""Risk and performance metrics for Market NN Plus Ultra predictions."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_equity_curve(returns: np.ndarray, initial_capital: float = 1.0) -> np.ndarray:
    """Return the cumulative equity curve for a sequence of returns."""

    return np.cumsum(returns, axis=-1) + initial_capital


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the (annualised) Sharpe ratio of the returns series."""

    return float(returns.mean() / (returns.std(ddof=0) + eps))


def sortino_ratio(returns: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Sortino ratio using downside deviation."""

    downside = returns[returns < 0]
    downside_std = np.sqrt((downside**2).mean() + eps)
    return float(returns.mean() / (downside_std + eps))


def max_drawdown(equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    """Return the maximum drawdown of the equity curve (negative value)."""

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / (running_max + eps)
    return float(drawdowns.min())


def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Calmar ratio using annualised return over max drawdown."""

    annualised_return = returns.mean() * 252
    max_dd = abs(max_drawdown(equity_curve, eps))
    return float(annualised_return / (max_dd + eps))


def hit_rate(returns: np.ndarray, eps: float = 1e-9) -> float:
    """Return the proportion of positive returns in the series."""

    if returns.size == 0:
        return 0.0
    positives = (returns > 0).sum()
    return float((positives + eps) / (returns.size + eps))


def downside_deviation(returns: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the downside deviation used in Sortino-like objectives."""

    downside = returns[returns < 0]
    if downside.size == 0:
        return 0.0
    return float(np.sqrt(((downside**2).mean()) + eps))


def ulcer_index(equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    """Return the Ulcer index, a severity-weighted drawdown measure."""

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / (running_max + eps)
    return float(np.sqrt(np.mean(drawdowns**2)))


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the one-sided Value-at-Risk at the given confidence level."""

    return float(np.quantile(returns, alpha))


def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Return the Conditional Value-at-Risk (Expected Shortfall)."""

    threshold = np.quantile(returns, alpha)
    tail = returns[returns <= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(tail.mean())


def risk_metrics(returns: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    """Return a suite of risk metrics for a stream of returns."""

    equity = compute_equity_curve(returns)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    drawdown = max_drawdown(equity)
    calmar = calmar_ratio(returns, equity)
    volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    downside_dev = downside_deviation(returns)
    hit = hit_rate(returns)
    ulcer = ulcer_index(equity)
    var = value_at_risk(returns)
    cvar = expected_shortfall(returns)
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "volatility": volatility,
        "downside_deviation": downside_dev,
        "hit_rate": hit,
        "ulcer_index": ulcer,
        "value_at_risk": var,
        "expected_shortfall": cvar,
    }


def evaluate_trade_log(trades: pd.DataFrame) -> Dict[str, float]:
    """Compute risk metrics for a trade log with a ``pnl`` column."""

    returns = trades["pnl"].to_numpy()
    return risk_metrics(returns)
