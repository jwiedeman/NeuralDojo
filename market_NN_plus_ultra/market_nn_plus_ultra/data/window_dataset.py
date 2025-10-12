"""Torch dataset utilities for sliding-window market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class WindowConfig:
    """Configuration for slicing time-series windows."""

    window_size: int = 256
    forecast_horizon: int = 5
    stride: int = 1
    target_column: str = "close"
    normalise: bool = True


class ZScoreNormalizer:
    """Normalise features using running mean and std per column."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None

    def fit(self, tensor: torch.Tensor) -> "ZScoreNormalizer":
        self.mean_ = tensor.mean(dim=(0, 1), keepdim=True)
        self.std_ = tensor.std(dim=(0, 1), keepdim=True).clamp_min(self.epsilon)
        return self

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer must be fitted before calling transform().")
        return (tensor - self.mean_) / self.std_

    def fit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fit(tensor).transform(tensor)


class SlidingWindowDataset(Dataset):
    """Convert a multi-indexed dataframe into model-ready windows."""

    def __init__(self, panel: pd.DataFrame, config: WindowConfig):
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("Panel must have a MultiIndex of (asset_id, timestamp)")

        self.config = config
        self.asset_ids = panel.index.get_level_values(0).unique()
        self.feature_columns = panel.columns.tolist()
        self.target_column = config.target_column
        try:
            self.target_idx = self.feature_columns.index(self.target_column)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Target column '{self.target_column}' not present in panel columns") from exc

        # Pre-compute windows per asset for efficient __getitem__.
        self.windows: list[tuple[int, int]] = []  # (asset_idx, start_position)
        self.asset_frames: list[pd.DataFrame] = []
        for asset_id in self.asset_ids:
            frame = panel.xs(asset_id).sort_index()
            self.asset_frames.append(frame)
            series_length = len(frame)
            max_start = series_length - (config.window_size + config.forecast_horizon)
            for start in range(0, max_start + 1, config.stride):
                self.windows.append((len(self.asset_frames) - 1, start))

        # Build tensor cache for all sequences to compute normalisation quickly.
        self.normalizer = ZScoreNormalizer() if config.normalise else None
        if self.normalizer is not None:
            tensor_bank = []
            for asset_index, start in self.windows:
                frame = self.asset_frames[asset_index].iloc[start : start + config.window_size]
                tensor_bank.append(torch.tensor(frame.values, dtype=torch.float32))
            stacked = torch.stack(tensor_bank)
            self.normalizer.fit(stacked)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        asset_index, start = self.windows[idx]
        frame = self.asset_frames[asset_index]
        window = frame.iloc[start : start + self.config.window_size]
        horizon = frame.iloc[
            start + self.config.window_size : start + self.config.window_size + self.config.forecast_horizon
        ]

        features = torch.tensor(window.values, dtype=torch.float32)
        if self.normalizer is not None:
            features = self.normalizer.transform(features.unsqueeze(0)).squeeze(0)

        targets = torch.tensor(horizon[self.target_column].values, dtype=torch.float32)
        return features, targets
