"""Risk metric utilities for reinforcement objectives and reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _validate_dim(tensor: torch.Tensor, dim: int) -> int:
    if tensor.ndim == 0:
        raise ValueError("risk metrics require at least one dimension")
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(f"dimension index {dim} is out of bounds for tensor with {tensor.ndim} dims")
    return dim


def sharpe_ratio(series: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Compute the Sharpe ratio for the supplied return series."""

    dim = _validate_dim(series, dim)
    mean = series.mean(dim=dim)
    std = series.std(dim=dim, unbiased=False)
    std = torch.clamp(std, min=eps)
    return mean / std


def sortino_ratio(series: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Compute the Sortino ratio for the supplied return series."""

    dim = _validate_dim(series, dim)
    downside = torch.where(series < 0, series, torch.zeros_like(series))
    downside_variance = (downside.pow(2)).mean(dim=dim)
    downside_std = torch.sqrt(downside_variance + eps)
    target_return = series.mean(dim=dim)
    return target_return / (downside_std + eps)


def max_drawdown(series: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Return the maximum drawdown magnitude for the return series."""

    dim = _validate_dim(series, dim)
    cumulative = series.cumsum(dim=dim)
    rolling_max, _ = torch.cummax(cumulative, dim=dim)
    drawdown = (cumulative - rolling_max) / (torch.abs(rolling_max) + eps)
    magnitude, _ = torch.min(drawdown, dim=dim)
    return torch.abs(magnitude)


def conditional_value_at_risk(
    series: torch.Tensor,
    *,
    alpha: float = 0.05,
    dim: int = -1,
) -> torch.Tensor:
    """Compute the Conditional Value at Risk (CVaR) for the series."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    dim = _validate_dim(series, dim)
    horizon = series.size(dim)
    if horizon == 0:
        raise ValueError("series must contain at least one element along the reduction dimension")

    sorted_returns, _ = torch.sort(series, dim=dim)
    tail_count = max(1, math.ceil(alpha * horizon))
    tail = sorted_returns.narrow(dim, 0, tail_count)
    return tail.mean(dim=dim)


@dataclass(slots=True)
class RiskMetrics:
    """Container holding standardised risk statistics."""

    sharpe: torch.Tensor
    sortino: torch.Tensor
    drawdown: torch.Tensor
    cvar: torch.Tensor


def compute_risk_metrics(
    series: torch.Tensor,
    *,
    alpha: float = 0.05,
    dim: int = -1,
    eps: float = 1e-6,
) -> RiskMetrics:
    """Convenience helper returning a bundle of risk metrics."""

    return RiskMetrics(
        sharpe=sharpe_ratio(series, dim=dim, eps=eps),
        sortino=sortino_ratio(series, dim=dim, eps=eps),
        drawdown=max_drawdown(series, dim=dim, eps=eps),
        cvar=conditional_value_at_risk(series, alpha=alpha, dim=dim),
    )


__all__ = [
    "RiskMetrics",
    "compute_risk_metrics",
    "conditional_value_at_risk",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
]
