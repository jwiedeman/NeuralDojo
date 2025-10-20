"""Risk-aware loss functions for Market NN Plus Ultra."""

from __future__ import annotations

import torch
from torch import nn

from ..trading.pnl import TradingCosts, differentiable_pnl, price_to_returns
from ..trading.risk import max_drawdown, sharpe_ratio, sortino_ratio


def sharpe_ratio_loss(returns: torch.Tensor) -> torch.Tensor:
    """Differentiable Sharpe ratio loss (negative Sharpe averaged over batch)."""

    sharpe = sharpe_ratio(returns, dim=-1)
    return -sharpe.mean()


def sortino_ratio_loss(returns: torch.Tensor) -> torch.Tensor:
    """Differentiable Sortino ratio loss averaged over the batch."""

    sortino = sortino_ratio(returns, dim=-1)
    return -sortino.mean()


def max_drawdown_penalty(returns: torch.Tensor) -> torch.Tensor:
    """Maximum drawdown penalty averaged over the batch."""

    drawdown = max_drawdown(returns, dim=-1)
    return drawdown.mean()


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

