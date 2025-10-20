import torch

from market_nn_plus_ultra.simulation import ExecutionConfig, simulate_execution


def test_partial_fill_and_positions():
    orders = torch.tensor([[[1.0], [1.0], [1.0]]])
    mid = torch.full_like(orders, 100.0)
    liquidity = torch.full_like(orders, 0.5)

    config = ExecutionConfig(
        base_slippage_bps=0.0,
        impact_slippage_bps=0.0,
        fee_bps=0.0,
        max_volume_fraction=1.0,
    )

    result = simulate_execution(orders, mid, liquidity, config=config)

    expected_half = orders.new_tensor([[0.5]])
    assert torch.allclose(result.executed[:, 0, :], expected_half)
    assert torch.allclose(result.unfilled[:, 0, :], expected_half)
    assert torch.allclose(result.positions[:, -1, :], orders.new_tensor([[1.5]]))
    assert torch.allclose(result.pnl[:, -1], orders.new_zeros(1))


def test_latency_penalty_increases_execution_price():
    orders = torch.tensor([[[1.0]], [[1.0]]])
    mid = torch.full_like(orders, 100.0)
    liquidity = torch.full_like(orders, 5.0)

    config = ExecutionConfig(
        base_slippage_bps=0.0,
        impact_slippage_bps=0.0,
        fee_bps=0.0,
        latency_buckets=(
            (0.2, 0.0),
            (0.5, 10.0),
        ),
    )

    latency = torch.tensor([[0.1], [0.5]])
    result = simulate_execution(orders, mid, liquidity, latency=latency, config=config)

    fast_price = result.avg_execution_price[0, 0, 0]
    slow_price = result.avg_execution_price[1, 0, 0]

    assert torch.isclose(fast_price, orders.new_tensor(100.0))
    assert slow_price > fast_price
    assert torch.isclose(result.latency_penalty_bps[1, 0], orders.new_tensor(10.0))


def test_funding_and_fees_reduce_pnl():
    orders = torch.tensor([[[1.0], [0.0]]])
    mid = torch.tensor([[[100.0], [105.0]]])
    liquidity = torch.full_like(orders, 5.0)

    config = ExecutionConfig(
        base_slippage_bps=0.0,
        impact_slippage_bps=0.0,
        fee_bps=5.0,
        funding_rate=0.001,
        max_volume_fraction=1.0,
    )

    result = simulate_execution(orders, mid, liquidity, config=config)

    assert torch.allclose(result.fee_cost[:, 0, :], orders.new_tensor([[0.05]]))
    assert torch.allclose(result.funding_cost[:, 1], orders.new_tensor([0.105]))
    expected_pnl = 5.0 - 0.05 - 0.105
    assert torch.isclose(result.final_pnl, orders.new_tensor([expected_pnl]), atol=1e-6)


def test_max_position_limit():
    orders = torch.tensor([[[1.0], [1.0]]])
    mid = torch.full_like(orders, 100.0)
    liquidity = torch.full_like(orders, 5.0)

    config = ExecutionConfig(
        base_slippage_bps=0.0,
        impact_slippage_bps=0.0,
        fee_bps=0.0,
        max_position=1.0,
    )

    result = simulate_execution(orders, mid, liquidity, config=config)

    assert torch.allclose(result.executed[:, 0, :], orders.new_tensor([[1.0]]))
    assert torch.allclose(result.executed[:, 1, :], orders.new_zeros((1, 1)))
    assert torch.allclose(result.unfilled[:, 1, :], orders.new_tensor([[1.0]]))
    assert torch.allclose(result.positions[:, -1, :], orders.new_tensor([[1.0]]))
