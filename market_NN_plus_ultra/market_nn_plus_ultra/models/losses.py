"""Risk-aware loss functions for Market NN Plus Ultra."""

from __future__ import annotations

import torch
from torch import nn

from ..trading.pnl import TradingCosts, differentiable_pnl, price_to_returns


def sharpe_ratio_loss(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Sharpe ratio loss (negative Sharpe)."""

    mean = returns.mean()
    std = returns.std(unbiased=False) + eps
    sharpe = mean / std
    return -sharpe


def sortino_ratio_loss(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    downside = returns[returns < 0]
    downside_std = downside.pow(2).mean().sqrt() + eps
    target_return = returns.mean()
    sortino = target_return / downside_std
    return -sortino


def max_drawdown_penalty(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cumulative = returns.cumsum(dim=-1)
    rolling_max = torch.cummax(cumulative, dim=-1)[0]
    drawdown = (cumulative - rolling_max) / (rolling_max.abs() + eps)
    return drawdown.abs().max()


class CompositeTradingLoss(nn.Module):
    """Combine regression, Sharpe, and drawdown penalties."""

    def __init__(
        self,
        sharpe_weight: float = 0.2,
        drawdown_weight: float = 0.1,
        mse_weight: float = 1.0,
        *,
        trading_costs: TradingCosts | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.trading_costs = trading_costs or TradingCosts()
        self.activation = activation

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mse = self.mse(preds, targets)
        future_returns = price_to_returns(targets, reference)
        pnl_series = differentiable_pnl(
            preds,
            future_returns,
            costs=self.trading_costs,
            activation=self.activation,
        )
        pnl_series = pnl_series.mean(dim=-1)
        sharpe = sharpe_ratio_loss(pnl_series)
        drawdown = max_drawdown_penalty(pnl_series)
        return self.mse_weight * mse + self.sharpe_weight * sharpe + self.drawdown_weight * drawdown


def composite_trading_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    return CompositeTradingLoss()(preds, targets, reference=reference)

