"""Pytest configuration for Market NN Plus Ultra.

Ensures the package can be imported without an editable install and
restores script module resolution for local test runs."""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve project and repository roots relative to this file.
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
_REPO_ROOT = _PROJECT_ROOT.parent

# Add roots to sys.path so ``import market_nn_plus_ultra`` and ``import scripts``
# succeed when running ``pytest`` from the repository without installing the
# package in editable mode.
for path in {_PROJECT_ROOT, _REPO_ROOT}:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
