"""Loss functions tailored for trading objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(slots=True)
class RiskAwareLoss:
    """Combines prediction error with risk-sensitive penalties."""

    base_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    sharpe_target: float = 2.0
    drawdown_lambda: float = 0.1

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, pnl: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid loss.

        Args:
            predictions: Model outputs (batch, horizon, output_dim).
            targets: Ground-truth labels, typically actions or returns.
            pnl: Simulated profit-and-loss trajectory aligned with predictions.
        """

        loss = self.base_loss(predictions, targets)
        excess = pnl - pnl.mean()
        sharpe = excess.mean() / (excess.std() + 1e-6)
        sharpe_penalty = torch.relu(self.sharpe_target - sharpe)
        drawdown = torch.cummax(-pnl, dim=-1)[0].max()
        return loss + self.drawdown_lambda * (sharpe_penalty + drawdown)


def default_risk_loss() -> RiskAwareLoss:
    """Return a default risk-aware loss instance using MSE base loss."""

    return RiskAwareLoss(base_loss=nn.MSELoss(reduction="mean"))
