"""Sliding window dataset utilities for temporal market modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class WindowSpec:
    """Specification for window extraction parameters."""

    window_size: int
    horizon: int
    stride: int = 1


@dataclass(slots=True)
class WindowMetadata:
    """Metadata describing a single sliding window sample."""

    symbol: str
    start_index: int
    input_timestamps: pd.Index
    target_timestamps: pd.Index


class SlidingWindowDataset(Dataset):
    """Turn a multi-indexed market panel into sliding windows."""

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_columns: Optional[Sequence[str]] = None,
        target_columns: Optional[Sequence[str]] = None,
        window_size: int = 256,
        horizon: int = 5,
        stride: int = 1,
        normalise: bool = True,
    ) -> None:
        if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["timestamp", "symbol"]:
            raise ValueError("Panel must be indexed by ('timestamp', 'symbol')")

        self.panel = panel
        self.feature_columns = list(feature_columns or [c for c in panel.columns if c not in ("symbol",)])
        self.target_columns = list(target_columns or ["close"])
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.normalise = normalise

        self._indices: list[Tuple[str, int]] = self._build_indices()

    def _build_indices(self) -> list[Tuple[str, int]]:
        indices: list[Tuple[str, int]] = []
        for symbol in self.panel.index.get_level_values("symbol").unique():
            sym_df = self.panel.xs(symbol, level="symbol")
            for start in range(0, len(sym_df) - self.window_size - self.horizon + 1, self.stride):
                indices.append((symbol, start))
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def _normalise(self, window: np.ndarray) -> np.ndarray:
        if not self.normalise:
            return window
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-6
        return (window - mean) / std

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        symbol, start = self._indices[idx]
        sym_df = self.panel.xs(symbol, level="symbol")

        window_slice = slice(start, start + self.window_size)
        target_slice = slice(start + self.window_size, start + self.window_size + self.horizon)

        window = sym_df.iloc[window_slice][self.feature_columns].to_numpy(dtype=np.float32)
        targets = sym_df.iloc[target_slice][self.target_columns].to_numpy(dtype=np.float32)

        window = self._normalise(window)

        return {
            "symbol": symbol,
            "features": torch.from_numpy(window),
            "targets": torch.from_numpy(targets),
        }

    def get_metadata(self, idx: int) -> WindowMetadata:
        """Return metadata for a specific window without disturbing training."""

        symbol, start = self._indices[idx]
        sym_df = self.panel.xs(symbol, level="symbol")
        window_slice = slice(start, start + self.window_size)
        target_slice = slice(start + self.window_size, start + self.window_size + self.horizon)
        return WindowMetadata(
            symbol=symbol,
            start_index=start,
            input_timestamps=sym_df.iloc[window_slice].index,
            target_timestamps=sym_df.iloc[target_slice].index,
        )

