"""Evaluation utilities for Market NN Plus Ultra."""

from .benchmarking import (
    ArchitectureSummary,
    format_markdown_table,
    load_benchmark_frames,
    summarise_architecture_performance,
    summaries_to_frame,
)
from .metrics import (
    calmar_ratio,
    compute_equity_curve,
    downside_deviation,
    evaluate_trade_log,
    expected_shortfall,
    guardrail_metrics,
    hit_rate,
    max_drawdown,
    risk_metrics,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
    value_at_risk,
)
from .operations import (
    OperationsSummary,
    OperationsThresholds,
    compile_operations_summary,
)
from .reporting import (
    MilestoneReference,
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
    "ArchitectureSummary",
    "format_markdown_table",
    "load_benchmark_frames",
    "summarise_architecture_performance",
    "summaries_to_frame",
    "calmar_ratio",
    "compute_equity_curve",
    "downside_deviation",
    "evaluate_trade_log",
    "expected_shortfall",
    "guardrail_metrics",
    "hit_rate",
    "max_drawdown",
    "OperationsSummary",
    "OperationsThresholds",
    "compile_operations_summary",
    "risk_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "ulcer_index",
    "value_at_risk",
    "MilestoneReference",
    "ReportSummary",
    "generate_markdown_report",
    "generate_html_report",
    "generate_report",
    "WalkForwardBacktester",
    "WalkForwardConfig",
    "WalkForwardSplit",
    "generate_walk_forward_splits",
]
