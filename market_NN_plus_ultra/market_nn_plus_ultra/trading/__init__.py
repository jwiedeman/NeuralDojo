"""Trading and inference utilities for Market NN Plus Ultra."""

from .pnl import TradingCosts, differentiable_pnl, price_to_returns

__all__ = [
    "MarketNNPlusUltraAgent",
    "AgentRunResult",
    "TradingCosts",
    "price_to_returns",
    "differentiable_pnl",
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
