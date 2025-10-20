"""Tests for risk-aware reinforcement objectives and metrics."""

from __future__ import annotations

import torch

from market_nn_plus_ultra.trading.risk import compute_risk_metrics
from market_nn_plus_ultra.training import RiskObjectiveConfig
from market_nn_plus_ultra.training.reinforcement import _apply_risk_objectives


def test_compute_risk_metrics_shapes() -> None:
    series = torch.tensor([[0.01, -0.02, 0.03], [0.0, 0.0, 0.0]], dtype=torch.float32)
    metrics = compute_risk_metrics(series, alpha=0.5)

    assert metrics.sharpe.shape == (2,)
    assert metrics.sortino.shape == (2,)
    assert metrics.drawdown.shape == (2,)
    assert metrics.cvar.shape == (2,)
    assert torch.all(metrics.drawdown >= 0)


def test_apply_risk_objectives_penalises_drawdown() -> None:
    pnl = torch.tensor([[0.02, -0.04, 0.01]], dtype=torch.float32)
    config = RiskObjectiveConfig(enabled=True, drawdown_weight=2.0, reward_scale=1.0)

    adjusted = _apply_risk_objectives(pnl, config)

    assert adjusted.shape == pnl.shape
    assert adjusted.mean() < pnl.mean()


def test_apply_risk_objectives_no_change_when_disabled() -> None:
    pnl = torch.tensor([[0.01, 0.0, -0.01]], dtype=torch.float32)
    config = RiskObjectiveConfig(enabled=False, drawdown_weight=5.0)

    adjusted = _apply_risk_objectives(pnl, config)

    torch.testing.assert_close(adjusted, pnl)
