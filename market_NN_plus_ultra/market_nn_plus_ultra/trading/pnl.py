"""Differentiable portfolio PnL utilities for Market NN Plus Ultra.

This module provides helpers that convert raw model actions into position
allocations and compute differentiable profit-and-loss streams that account for
transaction costs, slippage, and holding penalties.  The routines are designed
for use inside training losses so gradients can flow from ROI-style objectives
back into the model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import torch


ActivationFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class TradingCosts:
    """Cost parameters applied during PnL simulation."""

    transaction: float = 1e-4
    slippage: float = 1e-4
    holding: float = 0.0


def _resolve_activation(name: str | ActivationFn | None) -> ActivationFn:
    if name is None:
        return lambda x: x
    if callable(name):
        return name

    name = name.lower()
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return lambda x: torch.sigmoid(x) * 2.0 - 1.0
    if name in {"identity", "linear"}:
        return lambda x: x
    raise ValueError(f"Unknown activation '{name}' for position mapping")


def _expand_reference(reference: torch.Tensor | None, targets: torch.Tensor) -> torch.Tensor:
    """Return a tensor of reference prices aligned with *targets*.

    When *reference* is provided it is interpreted as the last observed price
    before the prediction horizon for each sample.  When omitted the first
    target is reused which implicitly sets the first return to zero.  This
    graceful fallback keeps training numerically stable even if raw price data
    is used as the target.
    """

    if reference is not None:
        if reference.dim() == targets.dim():
            base = reference.unsqueeze(1)
        else:
            base = reference
        if base.dim() != targets.dim():
            base = base.unsqueeze(1)
        return base

    # Fallback: treat the first future price as the starting reference.
    return targets[:, :1, :]


def price_to_returns(
    targets: torch.Tensor,
    reference: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert future price targets into log returns.

    Parameters
    ----------
    targets:
        Tensor shaped ``[batch, horizon, assets]`` with future prices or returns.
    reference:
        Optional tensor containing the last observed price before the horizon
        for each sample.  When omitted the first target price is reused which
        makes the first return zero.
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Tensor of log returns aligned with ``targets``.
    """

    if targets.ndim != 3:
        raise ValueError("targets must have shape [batch, horizon, assets]")

    base = _expand_reference(reference, targets)
    prices = torch.cat([base, targets], dim=1)
    prev = prices[:, :-1, :]
    nxt = prices[:, 1:, :]
    returns = torch.log((nxt + eps) / (prev + eps))
    return returns


def differentiable_pnl(
    actions: torch.Tensor,
    future_returns: torch.Tensor,
    *,
    costs: TradingCosts | Mapping[str, float] | None = None,
    activation: str | ActivationFn | None = "tanh",
    initial_position: float = 0.0,
) -> torch.Tensor:
    """Compute differentiable PnL for a batch of trajectories.

    Parameters
    ----------
    actions:
        Raw model outputs shaped ``[batch, horizon, assets]``.
    future_returns:
        Realised returns aligned with ``actions``.
    costs:
        Optional :class:`TradingCosts` or mapping containing ``transaction``,
        ``slippage``, and ``holding`` entries.
    activation:
        Name or callable used to map raw actions into position sizes.  The
        default ``"tanh"`` constrains leverage to ``[-1, 1]``.
    initial_position:
        Starting position for the first timestep.  A scalar value is broadcast
        across the batch.

    Returns
    -------
    torch.Tensor
        Per-sample PnL time-series shaped ``[batch, horizon]``.
    """

    if actions.shape != future_returns.shape:
        raise ValueError(
            "actions and future_returns must share the same shape; "
            f"got {actions.shape} vs {future_returns.shape}"
        )

    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, horizon, assets]")

    activation_fn = _resolve_activation(activation)
    positions = activation_fn(actions)

    if isinstance(costs, Mapping):
        costs = TradingCosts(**costs)  # type: ignore[arg-type]
    elif costs is None:
        costs = TradingCosts()

    batch, horizon, assets = positions.shape
    prev_position = torch.full(
        (batch, 1, assets),
        float(initial_position),
        device=positions.device,
        dtype=positions.dtype,
    )
    prev_position = torch.cat([prev_position, positions[:, :-1, :]], dim=1)

    gross = positions * future_returns
    turnover = torch.abs(positions - prev_position)
    transaction_penalty = costs.transaction * turnover
    slippage_penalty = costs.slippage * torch.abs(positions)
    holding_penalty = costs.holding * torch.abs(positions)

    pnl = gross - transaction_penalty - slippage_penalty - holding_penalty
    portfolio_pnl = pnl.sum(dim=-1)
    return portfolio_pnl


__all__ = [
    "TradingCosts",
    "price_to_returns",
    "differentiable_pnl",
]
