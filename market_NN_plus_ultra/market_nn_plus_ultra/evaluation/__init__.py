"""Evaluation utilities for Market NN Plus Ultra."""

from .metrics import (
    calmar_ratio,
    compute_equity_curve,
    downside_deviation,
    evaluate_trade_log,
    expected_shortfall,
    hit_rate,
    max_drawdown,
    risk_metrics,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
    value_at_risk,
)
from .reporting import (
    ReportSummary,
    generate_html_report,
    generate_markdown_report,
    generate_report,
)
from .walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardSplit,
    generate_walk_forward_splits,
)

__all__ = [
    "calmar_ratio",
    "compute_equity_curve",
    "downside_deviation",
    "evaluate_trade_log",
    "expected_shortfall",
    "hit_rate",
    "max_drawdown",
    "risk_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "ulcer_index",
    "value_at_risk",
    "ReportSummary",
    "generate_markdown_report",
    "generate_html_report",
    "generate_report",
    "WalkForwardBacktester",
    "WalkForwardConfig",
    "WalkForwardSplit",
    "generate_walk_forward_splits",
]
