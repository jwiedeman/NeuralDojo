"""Trading and inference utilities for Market NN Plus Ultra."""

from .guardrails import GuardrailConfig, GuardrailPolicy, GuardrailResult, GuardrailViolation
from .pnl import TradingCosts, differentiable_pnl, price_to_returns
from .risk import (
    RiskMetrics,
    compute_risk_metrics,
    conditional_value_at_risk,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

__all__ = [
    "MarketNNPlusUltraAgent",
    "AgentRunResult",
    "TradingCosts",
    "price_to_returns",
    "differentiable_pnl",
    "RiskMetrics",
    "compute_risk_metrics",
    "conditional_value_at_risk",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "GuardrailConfig",
    "GuardrailPolicy",
    "GuardrailResult",
    "GuardrailViolation",
]


def __getattr__(name: str):
    if name in {"MarketNNPlusUltraAgent", "AgentRunResult"}:
        from .agent import AgentRunResult, MarketNNPlusUltraAgent

        globals().update(
            {
                "MarketNNPlusUltraAgent": MarketNNPlusUltraAgent,
                "AgentRunResult": AgentRunResult,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'market_nn_plus_ultra.trading' has no attribute '{name}'")
