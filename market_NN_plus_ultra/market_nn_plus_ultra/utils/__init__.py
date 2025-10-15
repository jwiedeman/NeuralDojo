"""Utility helpers for Market NN Plus Ultra."""

from __future__ import annotations

import importlib
from typing import Any

from .logging import StructuredLogger, get_structured_logger
from .reporting import format_metrics_table, sanitize_metrics, write_metrics_report
from .seeding import seed_everything

__all__ = [
    "seed_everything",
    "format_metrics_table",
    "sanitize_metrics",
    "write_metrics_report",
    "StructuredLogger",
    "get_structured_logger",
    "ThroughputReport",
    "profile_backbone_throughput",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    if name in {"ThroughputReport", "profile_backbone_throughput"}:
        module = importlib.import_module(".profiling", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
