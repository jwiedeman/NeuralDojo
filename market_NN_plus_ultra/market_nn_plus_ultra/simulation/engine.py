"""Vectorised execution simulator with latency and funding effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


__all__ = [
    "ExecutionConfig",
    "ExecutionResult",
    "LatencyBucket",
    "simulate_execution",
]


@dataclass(slots=True)
class LatencyBucket:
    """Latency bucket definition expressed in seconds and basis points."""

    upper_bound: float
    penalty_bps: float

    def penalty_decimal(self) -> float:
        return self.penalty_bps / 1e4


@dataclass(slots=True)
class ExecutionConfig:
    """Configuration controlling simulator realism assumptions."""

    base_slippage_bps: float = 1.0
    impact_slippage_bps: float = 5.0
    fee_bps: float = 0.5
    funding_rate: float = 0.0
    max_volume_fraction: float = 1.0
    latency_buckets: Sequence[LatencyBucket | tuple[float, float]] = (
        LatencyBucket(upper_bound=0.25, penalty_bps=0.0),
        LatencyBucket(upper_bound=0.5, penalty_bps=2.0),
        LatencyBucket(upper_bound=1.0, penalty_bps=5.0),
    )
    max_position: float | None = None

    def __post_init__(self) -> None:
        buckets: list[LatencyBucket] = []
        for bucket in self.latency_buckets:
            if isinstance(bucket, LatencyBucket):
                buckets.append(bucket)
            else:
                upper, penalty = bucket
                buckets.append(LatencyBucket(upper_bound=float(upper), penalty_bps=float(penalty)))
        buckets.sort(key=lambda bucket: bucket.upper_bound)
        object.__setattr__(self, "latency_buckets", tuple(buckets))

    @property
    def base_slippage(self) -> float:
        return self.base_slippage_bps / 1e4

    @property
    def impact_slippage(self) -> float:
        return self.impact_slippage_bps / 1e4

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 1e4

    def latency_penalty(self, latencies: torch.Tensor) -> torch.Tensor:
        """Return per-sample latency penalties in decimal form."""

        if not self.latency_buckets:
            return torch.zeros_like(latencies)

        penalty = torch.zeros_like(latencies)
        remaining = torch.ones_like(latencies, dtype=torch.bool)
        for bucket in self.latency_buckets:
            mask = remaining & (latencies <= bucket.upper_bound)
            if mask.any():
                penalty = torch.where(
                    mask,
                    torch.as_tensor(
                        bucket.penalty_decimal(),
                        dtype=latencies.dtype,
                        device=latencies.device,
                    ),
                    penalty,
                )
                remaining = remaining & (~mask)
        if remaining.any():
            last = self.latency_buckets[-1].penalty_decimal()
            penalty = torch.where(
                remaining,
                torch.as_tensor(last, dtype=latencies.dtype, device=latencies.device),
                penalty,
            )
        return penalty


@dataclass(slots=True)
class ExecutionResult:
    """Structured return payload from :func:`simulate_execution`."""

    executed: torch.Tensor
    unfilled: torch.Tensor
    positions: torch.Tensor
    cash: torch.Tensor
    pnl: torch.Tensor
    avg_execution_price: torch.Tensor
    slippage_cost: torch.Tensor
    fee_cost: torch.Tensor
    funding_cost: torch.Tensor
    latency_penalty_bps: torch.Tensor

    @property
    def final_position(self) -> torch.Tensor:
        return self.positions[:, -1, :]

    @property
    def final_cash(self) -> torch.Tensor:
        return self.cash[:, -1]

    @property
    def final_pnl(self) -> torch.Tensor:
        return self.pnl[:, -1]


def _ensure_tensor(
    value: torch.Tensor | float | Sequence[float],
    *,
    shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim > len(shape):
        raise ValueError(
            f"Value with shape {tuple(tensor.shape)} cannot broadcast to {tuple(shape)}"
        )
    while tensor.ndim < len(shape):
        tensor = tensor.unsqueeze(0)
    try:
        return tensor.expand(shape)
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Value with shape {tuple(tensor.shape)} cannot broadcast to {tuple(shape)}"
        ) from exc


def simulate_execution(
    orders: torch.Tensor,
    mid_prices: torch.Tensor,
    available_liquidity: torch.Tensor,
    *,
    spread: torch.Tensor | float | Sequence[float] | None = None,
    latency: torch.Tensor | float | Sequence[float] | None = None,
    config: ExecutionConfig | None = None,
    dt: float = 1.0,
) -> ExecutionResult:
    """Simulate execution with partial fills, slippage, and funding."""

    if config is None:
        config = ExecutionConfig()

    if orders.ndim == 2:
        orders = orders.unsqueeze(-1)
        mid_prices = mid_prices.unsqueeze(-1)
        available_liquidity = available_liquidity.unsqueeze(-1)
        if isinstance(spread, torch.Tensor) and spread.ndim == 2:
            spread = spread.unsqueeze(-1)

    if orders.shape != mid_prices.shape:
        raise ValueError("orders and mid_prices must have identical shapes")
    if available_liquidity.shape != orders.shape:
        raise ValueError("available_liquidity must match orders shape")

    batch, steps, assets = orders.shape
    device = orders.device
    dtype = orders.dtype

    if spread is None:
        spread_tensor = torch.zeros((batch, steps, assets), device=device, dtype=dtype)
    else:
        spread_tensor = _ensure_tensor(spread, shape=(batch, steps, assets), device=device, dtype=dtype)

    if latency is None:
        latency_tensor = torch.zeros((batch, steps), device=device, dtype=dtype)
    else:
        latency_tensor = _ensure_tensor(latency, shape=(batch, steps), device=device, dtype=dtype)

    latency_penalty = config.latency_penalty(latency_tensor)
    latency_penalty = latency_penalty.unsqueeze(-1)  # align with assets

    capacity = torch.clamp(available_liquidity, min=0.0) * float(config.max_volume_fraction)
    eps = torch.finfo(dtype).eps

    executed_records = []
    unfilled_records = []
    position_records = []
    cash_records = []
    pnl_records = []
    price_records = []
    slippage_records = []
    fee_records = []
    funding_records = []
    penalty_records = []

    position = torch.zeros((batch, assets), device=device, dtype=dtype)
    cash = torch.zeros(batch, device=device, dtype=dtype)

    base_slippage = config.base_slippage
    impact_slippage = config.impact_slippage
    fee_rate = config.fee_rate
    funding_rate = float(config.funding_rate)

    for t in range(steps):
        desired = orders[:, t, :]
        price = mid_prices[:, t, :]
        avail = capacity[:, t, :]
        prev_position = position

        exec_abs = torch.minimum(desired.abs(), avail)
        executed = exec_abs * torch.sign(desired)

        if config.max_position is not None:
            max_pos = float(config.max_position)
            target_position = torch.clamp(prev_position + executed, -max_pos, max_pos)
            executed = target_position - prev_position
            exec_abs = executed.abs()

        fill_ratio = exec_abs / (desired.abs() + eps)
        fill_ratio = torch.where(desired.abs() > eps, fill_ratio, torch.zeros_like(fill_ratio))

        total_slippage = base_slippage + impact_slippage * fill_ratio + latency_penalty[:, t, :]
        half_spread = 0.5 * spread_tensor[:, t, :]
        direction = torch.sign(executed)
        price_multiplier = 1.0 + direction * (half_spread + total_slippage)
        execution_price = price * price_multiplier

        slippage_cost = exec_abs * price * total_slippage
        fee_cost = exec_abs * price * fee_rate

        trade_cash = (executed * execution_price).sum(dim=-1)

        funding_cost = (prev_position * price * funding_rate * dt).sum(dim=-1)

        cash = cash - trade_cash - fee_cost.sum(dim=-1) - funding_cost
        position = prev_position + executed

        mark_to_market = (position * price).sum(dim=-1)
        pnl = cash + mark_to_market

        unfilled = desired - executed

        executed_records.append(executed)
        unfilled_records.append(unfilled)
        position_records.append(position)
        cash_records.append(cash)
        pnl_records.append(pnl)
        price_records.append(execution_price)
        slippage_records.append(slippage_cost)
        fee_records.append(fee_cost)
        funding_records.append(funding_cost)
        penalty_records.append(latency_penalty[:, t])

    executed_tensor = torch.stack(executed_records, dim=1)
    unfilled_tensor = torch.stack(unfilled_records, dim=1)
    positions_tensor = torch.stack(position_records, dim=1)
    cash_tensor = torch.stack(cash_records, dim=1)
    pnl_tensor = torch.stack(pnl_records, dim=1)
    price_tensor = torch.stack(price_records, dim=1)
    slippage_tensor = torch.stack(slippage_records, dim=1)
    fee_tensor = torch.stack(fee_records, dim=1)
    funding_tensor = torch.stack(funding_records, dim=1)
    penalty_tensor = torch.stack(penalty_records, dim=1)

    return ExecutionResult(
        executed=executed_tensor,
        unfilled=unfilled_tensor,
        positions=positions_tensor,
        cash=cash_tensor,
        pnl=pnl_tensor,
        avg_execution_price=price_tensor,
        slippage_cost=slippage_tensor,
        fee_cost=fee_tensor,
        funding_cost=funding_tensor,
        latency_penalty_bps=penalty_tensor * 1e4,
    )
