from __future__ import annotations

import torch

from market_nn_plus_ultra.trading import TradingCosts, differentiable_pnl, price_to_returns


def test_price_to_returns_with_reference() -> None:
    prices = torch.tensor(
        [
            [[102.0], [103.0], [104.0]],
            [[50.0], [49.0], [48.0]],
        ],
        dtype=torch.float32,
    )
    reference = torch.tensor([[101.0], [51.0]], dtype=torch.float32)
    returns = price_to_returns(prices, reference)
    assert returns.shape == prices.shape
    first_sample = returns[0].exp() - 1.0
    second_sample = returns[1].exp() - 1.0
    assert torch.allclose(first_sample[:, 0], torch.tensor([0.009900, 0.009804, 0.009709]), atol=1e-4)
    assert torch.allclose(second_sample[:, 0], torch.tensor([-0.019608, -0.020000, -0.020408]), atol=1e-4)


def test_differentiable_pnl_cost_penalties() -> None:
    actions = torch.tensor(
        [
            [[0.5], [0.5], [0.5]],
            [[-0.25], [0.25], [0.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    future_returns = torch.tensor(
        [
            [[0.02], [0.01], [-0.005]],
            [[-0.01], [0.015], [0.0]],
        ],
        dtype=torch.float32,
    )
    costs = TradingCosts(transaction=0.001, slippage=0.002, holding=0.0005)
    pnl = differentiable_pnl(actions, future_returns, costs=costs, activation="tanh")
    assert pnl.shape == torch.Size([2, 3])
    pnl.sum().backward()
    assert actions.grad is not None
    assert torch.all(torch.isfinite(actions.grad))
