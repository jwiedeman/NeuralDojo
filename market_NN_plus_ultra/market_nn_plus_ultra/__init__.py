"""Market NN Plus Ultra package with lazy-loaded submodules."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "data",
    "models",
    "trading",
    "training",
    "evaluation",
    "utils",
    "cli",
    "simulation",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple passthrough
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
