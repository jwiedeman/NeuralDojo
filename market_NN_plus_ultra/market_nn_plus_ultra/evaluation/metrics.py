"""Risk and performance metrics for trading strategies."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_equity_curve(returns: np.ndarray, initial_capital: float = 1.0) -> np.ndarray:
    equity = np.cumsum(returns, axis=-1) + initial_capital
    return equity


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-6) -> float:
    return float(returns.mean() / (returns.std(ddof=0) + eps))


def sortino_ratio(returns: np.ndarray, eps: float = 1e-6) -> float:
    downside = returns[returns < 0]
    downside_std = np.sqrt((downside ** 2).mean() + eps)
    return float(returns.mean() / (downside_std + eps))


def max_drawdown(equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    cumulative = equity_curve
    running_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - running_max) / (running_max + eps)
    return float(dd.min())


def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    annualised_return = returns.mean() * 252
    max_dd = abs(max_drawdown(equity_curve, eps))
    return float(annualised_return / (max_dd + eps))


def risk_metrics(returns: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    equity = compute_equity_curve(returns)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    drawdown = max_drawdown(equity)
    calmar = calmar_ratio(returns, equity)
    volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "volatility": volatility,
    }


def evaluate_trade_log(trades: pd.DataFrame) -> Dict[str, float]:
    returns = trades["pnl"].to_numpy()
    return risk_metrics(returns)

