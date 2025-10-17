"""Sliding window dataset utilities for temporal market modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _nan_to_num(array: np.ndarray) -> np.ndarray:
    """Return a copy of *array* with NaN/Inf values replaced by safe defaults."""

    if np.isfinite(array).all():
        return array.astype(np.float32, copy=False)

    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32
    zero = dtype.type(0.0)
    filled = np.nan_to_num(array, nan=zero, posinf=zero, neginf=zero)
    return np.asarray(filled, dtype=dtype)


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


@dataclass(slots=True)
class _SymbolData:
    """Cached per-symbol arrays to avoid repeated pandas slicing."""

    features: np.ndarray
    targets: np.ndarray
    index: pd.Index


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

        self._symbol_data: dict[str, _SymbolData] = self._precompute_symbol_data()
        self._indices: list[Tuple[str, int]] = self._build_indices()

    def _precompute_symbol_data(self) -> dict[str, _SymbolData]:
        symbol_data: dict[str, _SymbolData] = {}
        for symbol in self.panel.index.get_level_values("symbol").unique():
            sym_df = self.panel.xs(symbol, level="symbol")
            feature_array = sym_df[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            target_array = sym_df[self.target_columns].to_numpy(dtype=np.float32, copy=True)
            symbol_data[symbol] = _SymbolData(
                features=feature_array,
                targets=target_array,
                index=sym_df.index,
            )
        return symbol_data

    def _build_indices(self) -> list[Tuple[str, int]]:
        indices: list[Tuple[str, int]] = []
        for symbol, data in self._symbol_data.items():
            max_start = data.features.shape[0] - self.window_size - self.horizon + 1
            if max_start <= 0:
                continue
            for start in range(0, max_start, self.stride):
                indices.append((symbol, start))
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def _normalise(self, window: np.ndarray) -> np.ndarray:
        if not self.normalise:
            return _nan_to_num(window)

        clean = np.array(window, copy=True, dtype=np.float32)
        mask = np.isfinite(clean)

        if not mask.any():
            return np.zeros_like(clean, dtype=np.float32)

        safe_counts = mask.sum(axis=0, keepdims=True)
        safe_counts = np.maximum(safe_counts, 1)

        sums = np.where(mask, clean, 0.0).sum(axis=0, keepdims=True)
        mean = sums / safe_counts

        centred = np.where(mask, clean - mean, 0.0)
        sq_sums = np.square(centred).sum(axis=0, keepdims=True)
        variance = sq_sums / safe_counts
        std = np.sqrt(variance).astype(np.float32, copy=False)
        column_mask = mask.any(axis=0, keepdims=True)
        std = np.where(column_mask, std, 1.0).astype(np.float32, copy=False)
        std = np.maximum(std, 1e-6)

        filled = np.where(mask, clean, mean)
        normalised = (filled - mean) / std
        return _nan_to_num(normalised)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        symbol, start = self._indices[idx]
        data = self._symbol_data[symbol]

        window_slice = slice(start, start + self.window_size)
        target_slice = slice(start + self.window_size, start + self.window_size + self.horizon)

        window = data.features[window_slice]
        targets = data.targets[target_slice]

        window = self._normalise(window).astype(np.float32, copy=False)
        targets = _nan_to_num(targets).astype(np.float32, copy=False)

        reference = data.targets[start + self.window_size - 1]
        reference = _nan_to_num(reference).astype(np.float32, copy=False)
        return {
            "symbol": symbol,
            "features": torch.from_numpy(window.copy()),
            "targets": torch.from_numpy(targets.copy()),
            "reference": torch.from_numpy(reference.copy()),
        }

    def get_metadata(self, idx: int) -> WindowMetadata:
        """Return metadata for a specific window without disturbing training."""

        symbol, start = self._indices[idx]
        data = self._symbol_data[symbol]
        window_slice = slice(start, start + self.window_size)
        target_slice = slice(start + self.window_size, start + self.window_size + self.horizon)
        return WindowMetadata(
            symbol=symbol,
            start_index=start,
            input_timestamps=data.index[window_slice],
            target_timestamps=data.index[target_slice],
        )

