"""Risk and performance metrics for the Market NN Plus Ultra stack."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _coerce_returns(array: np.ndarray | pd.Series) -> np.ndarray:
    """Return a 1-D ``float64`` array with NaNs removed."""

    returns = np.asarray(array, dtype=np.float64).reshape(-1)
    if returns.size == 0:
        return returns
    return returns[~np.isnan(returns)]


def compute_equity_curve(returns: np.ndarray, initial_capital: float = 1.0) -> np.ndarray:
    """Return the cumulative equity curve for a sequence of returns."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return np.asarray([initial_capital], dtype=np.float64)
    cumulative = initial_capital + np.cumsum(cleaned, axis=-1)
    return np.concatenate((np.asarray([initial_capital], dtype=np.float64), cumulative))


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the (annualised) Sharpe ratio of the returns series."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    std = cleaned.std(ddof=0)
    if std <= eps:
        return 0.0
    return float(cleaned.mean() / (std + eps))


def sortino_ratio(returns: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Sortino ratio using downside deviation."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    downside = cleaned[cleaned < 0]
    if downside.size == 0:
        return 0.0
    downside_std = np.sqrt((downside**2).mean() + eps)
    return float(cleaned.mean() / (downside_std + eps))


def max_drawdown(equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    """Return the maximum drawdown of the equity curve (negative value)."""

    equity_curve = np.asarray(equity_curve, dtype=np.float64).reshape(-1)
    if equity_curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / (running_max + eps)
    return float(drawdowns.min())


def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Calmar ratio using annualised return over max drawdown."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    annualised_return = cleaned.mean() * 252
    max_dd = abs(max_drawdown(equity_curve, eps))
    if max_dd <= eps:
        return float("inf" if annualised_return > 0 else 0.0)
    return float(annualised_return / (max_dd + eps))


def hit_rate(returns: np.ndarray, eps: float = 1e-9) -> float:
    """Return the proportion of positive returns in the series."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    positives = (cleaned > 0).sum()
    return float((positives + eps) / (cleaned.size + eps))


def downside_deviation(returns: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the downside deviation used in Sortino-like objectives."""

    cleaned = _coerce_returns(returns)
    downside = cleaned[cleaned < 0]
    if downside.size == 0:
        return 0.0
    return float(np.sqrt(((downside**2).mean()) + eps))


def ulcer_index(equity_curve: np.ndarray, eps: float = 1e-6) -> float:
    """Return the Ulcer index, a severity-weighted drawdown measure."""

    equity_curve = np.asarray(equity_curve, dtype=np.float64).reshape(-1)
    if equity_curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / (running_max + eps)
    return float(np.sqrt(np.mean(drawdowns**2)))


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the one-sided Value-at-Risk at the given confidence level."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    return float(np.quantile(cleaned, alpha))


def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Return the Conditional Value-at-Risk (Expected Shortfall)."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    threshold = np.quantile(cleaned, alpha)
    tail = cleaned[cleaned <= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(tail.mean())


def omega_ratio(returns: np.ndarray, threshold: float = 0.0, eps: float = 1e-6) -> float:
    """Compute the Omega ratio relative to a return threshold."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    gains = np.maximum(cleaned - threshold, 0)
    losses = np.maximum(threshold - cleaned, 0)
    gain_sum = gains.sum()
    loss_sum = losses.sum()
    if loss_sum <= eps:
        return float("inf" if gain_sum > 0 else 0.0)
    return float(gain_sum / (loss_sum + eps))


def profit_factor(returns: np.ndarray, eps: float = 1e-6) -> float:
    """Return the ratio of gross profits to gross losses."""

    cleaned = _coerce_returns(returns)
    if cleaned.size == 0:
        return 0.0
    positive = cleaned[cleaned > 0].sum()
    negative = -cleaned[cleaned < 0].sum()
    if negative <= eps:
        return float("inf" if positive > 0 else 0.0)
    return float(positive / (negative + eps))


def risk_metrics(returns: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    """Return a suite of risk metrics for a stream of returns."""

    cleaned = _coerce_returns(returns)
    equity = compute_equity_curve(cleaned)
    sharpe = sharpe_ratio(cleaned)
    sortino = sortino_ratio(cleaned)
    drawdown = max_drawdown(equity)
    calmar = calmar_ratio(cleaned, equity)
    volatility = float(cleaned.std(ddof=0) * np.sqrt(periods_per_year)) if cleaned.size else 0.0
    downside_dev = downside_deviation(cleaned)
    hit = hit_rate(cleaned)
    ulcer = ulcer_index(equity)
    var = value_at_risk(cleaned)
    cvar = expected_shortfall(cleaned)
    omega = omega_ratio(cleaned)
    pf = profit_factor(cleaned)
    if equity.size >= 2:
        cumulative_return = float(equity[-1] - equity[0])
        if equity[0] != 0:
            total_return = float((equity[-1] / equity[0]) - 1.0)
            return_multiple = float(equity[-1] / equity[0])
        else:
            total_return = float("inf" if equity[-1] > 0 else 0.0)
            return_multiple = float("inf" if equity[-1] > 0 else 0.0)
    else:
        cumulative_return = 0.0
        total_return = 0.0
        return_multiple = 1.0
    average_return = float(cleaned.mean()) if cleaned.size else 0.0
    annualised_return = float(average_return * periods_per_year)
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
        "omega_ratio": omega,
        "profit_factor": pf,
        "cumulative_return": cumulative_return,
        "total_return": total_return,
        "return_multiple": return_multiple,
        "average_return": average_return,
        "annualised_return": annualised_return,
    }


def evaluate_trade_log(trades: pd.DataFrame) -> Dict[str, float]:
    """Compute risk metrics for a trade log with a ``pnl`` column."""

    returns = trades["pnl"].to_numpy(dtype=np.float64)
    return risk_metrics(returns)


def guardrail_metrics(
    trades: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    notional_col: str = "notional",
    position_col: str = "position",
    price_col: str = "price",
    return_col: str = "pnl",
    capital_base: float = 1.0,
    tail_percentile: float = 5.0,
) -> Dict[str, float]:
    """Compute guardrail diagnostics for a trade log.

    Parameters
    ----------
    trades:
        DataFrame containing at least a timestamp column and either a
        notional exposure column or both position and price columns.
    timestamp_col, symbol_col:
        Column names describing trade timestamps and symbols.
    notional_col, position_col, price_col:
        Exposure column names. When ``notional_col`` is missing the notional is
        derived as ``position * price``.
    return_col:
        Column describing realised returns or PnL for tail analysis.
    capital_base:
        Reference capital used to normalise exposure and turnover metrics.
    tail_percentile:
        Percentile (0–100) used to compute downside tail statistics.

    Returns
    -------
    dict
        Dictionary containing exposure, turnover, and tail risk diagnostics.
    """

    if timestamp_col not in trades:
        raise ValueError(f"Column '{timestamp_col}' is required for guardrail metrics")

    df = trades.copy()
    if notional_col not in df.columns:
        if position_col not in df.columns or price_col not in df.columns:
            raise ValueError(
                "Guardrail metrics require either a notional column or both"
                f" position ('{position_col}') and price ('{price_col}') columns"
            )
        df[notional_col] = df[position_col] * df[price_col]

    df = df.dropna(subset=[timestamp_col])
    if df.empty:
        return {
            "gross_exposure_peak": 0.0,
            "gross_exposure_mean": 0.0,
            "net_exposure_mean": 0.0,
            "net_exposure_peak": 0.0,
            "turnover_rate": 0.0,
            "tail_return_quantile": 0.0,
            "tail_return_mean": 0.0,
            "tail_event_frequency": 0.0,
            "max_symbol_exposure": 0.0,
        }

    df = df.sort_values(timestamp_col)
    capital = float(capital_base) if capital_base not in (None, 0) else 1.0

    grouped = df.groupby(timestamp_col)
    gross_series = grouped[notional_col].apply(lambda values: np.abs(values).sum())
    net_series = grouped[notional_col].sum()

    gross_peak = float(gross_series.max() / capital) if not gross_series.empty else 0.0
    gross_mean = float(gross_series.mean() / capital) if not gross_series.empty else 0.0
    net_mean = float(net_series.mean() / capital) if not net_series.empty else 0.0
    net_peak = float(net_series.abs().max() / capital) if not net_series.empty else 0.0

    if len(net_series) > 1:
        turnover = float(net_series.diff().abs().fillna(0.0).mean() / capital)
    else:
        turnover = 0.0

    tail_cut = 0.0
    tail_mean = 0.0
    tail_frequency = 0.0
    if return_col in df.columns:
        returns_by_time = grouped[return_col].sum()
        if not returns_by_time.empty:
            tail_cut = float(
                np.percentile(
                    returns_by_time.to_numpy(dtype=np.float64),
                    float(np.clip(tail_percentile, 0.0, 100.0)),
                )
            )
            tail_values = returns_by_time[returns_by_time <= tail_cut]
            tail_mean = float(tail_values.mean()) if not tail_values.empty else tail_cut
            tail_frequency = float(len(tail_values) / len(returns_by_time))

    max_symbol_exposure = 0.0
    if symbol_col in df.columns:
        symbol_exposure = df.groupby(symbol_col)[notional_col].apply(lambda values: np.abs(values).max())
        if not symbol_exposure.empty:
            max_symbol_exposure = float(symbol_exposure.max() / capital)

    return {
        "gross_exposure_peak": gross_peak,
        "gross_exposure_mean": gross_mean,
        "net_exposure_mean": net_mean,
        "net_exposure_peak": net_peak,
        "turnover_rate": turnover,
        "tail_return_quantile": tail_cut,
        "tail_return_mean": tail_mean,
        "tail_event_frequency": tail_frequency,
        "max_symbol_exposure": max_symbol_exposure,
    }
