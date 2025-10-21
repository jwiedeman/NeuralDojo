"""Helpers for reinforcement-learning CLI overrides."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

from ..trading import TradingCosts
from ..training import ReinforcementConfig

_COST_FIELDS = ("cost_transaction", "cost_slippage", "cost_holding")
_RISK_FIELDS = (
    "risk_enabled",
    "risk_sharpe_weight",
    "risk_sortino_weight",
    "risk_drawdown_weight",
    "risk_cvar_weight",
    "risk_reward_scale",
    "risk_cvar_alpha",
)


def register_reinforcement_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments used to tune reinforcement fine-tuning runs."""

    parser.add_argument("--activation", type=str, help="Activation used to map policy outputs into positions")
    parser.add_argument(
        "--cost-transaction",
        dest="cost_transaction",
        type=float,
        help="Transaction cost penalty applied to turnover",
    )
    parser.add_argument(
        "--cost-slippage",
        dest="cost_slippage",
        type=float,
        help="Slippage cost penalty applied to absolute positions",
    )
    parser.add_argument(
        "--cost-holding",
        dest="cost_holding",
        type=float,
        help="Holding cost penalty applied to absolute positions",
    )
    parser.add_argument(
        "--risk-enabled",
        dest="risk_enabled",
        action="store_true",
        help="Enable risk-aware reward shaping overrides",
    )
    parser.add_argument(
        "--risk-disabled",
        dest="risk_enabled",
        action="store_false",
        help="Disable risk-aware reward shaping overrides",
    )
    parser.set_defaults(risk_enabled=None)
    parser.add_argument("--risk-sharpe-weight", type=float, help="Sharpe ratio weight for reward shaping")
    parser.add_argument("--risk-sortino-weight", type=float, help="Sortino ratio weight for reward shaping")
    parser.add_argument("--risk-drawdown-weight", type=float, help="Drawdown penalty weight for reward shaping")
    parser.add_argument("--risk-cvar-weight", type=float, help="CVaR penalty weight for reward shaping")
    parser.add_argument("--risk-reward-scale", type=float, help="Scaling factor applied to risk adjustments")
    parser.add_argument("--risk-cvar-alpha", type=float, help="Alpha quantile used when computing CVaR penalties")
    parser.add_argument(
        "--targets-are-returns",
        action="store_true",
        help="Treat dataset targets as pre-computed returns instead of prices",
    )


def _any_override(args: argparse.Namespace, fields: Iterable[str]) -> bool:
    return any(getattr(args, field, None) is not None for field in fields)


def apply_reinforcement_overrides(
    config: ReinforcementConfig, args: argparse.Namespace
) -> ReinforcementConfig:
    """Apply trading-cost and risk overrides to a reinforcement config."""

    updated = replace(config)

    if getattr(args, "activation", None) is not None:
        updated.activation = args.activation
    if getattr(args, "targets_are_returns", False):
        updated.targets_are_returns = True

    if _any_override(args, _COST_FIELDS):
        base_costs = updated.costs
        costs = replace(base_costs) if base_costs is not None else TradingCosts()
        if args.cost_transaction is not None:
            costs.transaction = args.cost_transaction
        if args.cost_slippage is not None:
            costs.slippage = args.cost_slippage
        if args.cost_holding is not None:
            costs.holding = args.cost_holding
        updated.costs = costs

    if _any_override(args, _RISK_FIELDS):
        risk = replace(updated.risk_objective)
        if args.risk_enabled is not None:
            risk.enabled = args.risk_enabled
        if args.risk_sharpe_weight is not None:
            risk.sharpe_weight = args.risk_sharpe_weight
        if args.risk_sortino_weight is not None:
            risk.sortino_weight = args.risk_sortino_weight
        if args.risk_drawdown_weight is not None:
            risk.drawdown_weight = args.risk_drawdown_weight
        if args.risk_cvar_weight is not None:
            risk.cvar_weight = args.risk_cvar_weight
        if args.risk_reward_scale is not None:
            risk.reward_scale = args.risk_reward_scale
        if args.risk_cvar_alpha is not None:
            risk.cvar_alpha = args.risk_cvar_alpha
        updated.risk_objective = risk

    return updated


__all__ = ["apply_reinforcement_overrides", "register_reinforcement_arguments"]
