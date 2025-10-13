"""Walk-forward backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd

from .metrics import risk_metrics


@dataclass(slots=True)
class WalkForwardSplit:
    """Metadata describing a single walk-forward split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(slots=True)
class WalkForwardConfig:
    """Configuration for running walk-forward backtests."""

    train_window: int
    test_window: int
    step: int | None = None
    timestamp_column: str = "window_end"
    return_column: str = "realised_return"
    min_train_size: int = 1


def generate_walk_forward_splits(
    timestamps: Sequence[pd.Timestamp],
    train_window: int,
    test_window: int,
    step: int | None = None,
) -> List[WalkForwardSplit]:
    """Return a list of contiguous walk-forward splits.

    Parameters
    ----------
    timestamps:
        Ordered timestamps representing the evaluation index.
    train_window:
        Number of periods allocated to the training segment.
    test_window:
        Number of periods allocated to the test segment.
    step:
        Advance the window by this many periods after each split. Defaults to
        ``test_window`` (non-overlapping).
    """

    if train_window <= 0:
        raise ValueError("train_window must be positive")
    if test_window <= 0:
        raise ValueError("test_window must be positive")
    if step is None:
        step = test_window
    if step <= 0:
        raise ValueError("step must be positive")

    ordered = pd.Index(pd.to_datetime(list(timestamps))).sort_values()
    splits: List[WalkForwardSplit] = []
    for start in range(0, len(ordered) - train_window - test_window + 1, step):
        train_slice = ordered[start : start + train_window]
        test_slice = ordered[start + train_window : start + train_window + test_window]
        if train_slice.empty or test_slice.empty:
            continue
        splits.append(
            WalkForwardSplit(
                train_start=train_slice[0],
                train_end=train_slice[-1],
                test_start=test_slice[0],
                test_end=test_slice[-1],
            )
        )
    return splits


class WalkForwardBacktester:
    """Evaluate realised returns across walk-forward splits."""

    def __init__(
        self,
        config: WalkForwardConfig,
        metrics_fn: Callable[[np.ndarray], dict[str, float]] | None = None,
    ) -> None:
        if config.train_window < config.min_train_size:
            raise ValueError(
                "train_window is smaller than min_train_size; increase train_window or adjust min_train_size"
            )
        self.config = config
        self.metrics_fn = metrics_fn or risk_metrics

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return metrics per split for the supplied prediction frame."""

        if self.config.timestamp_column not in frame:
            raise ValueError(
                f"timestamp column '{self.config.timestamp_column}' not found in predictions frame"
            )
        if self.config.return_column not in frame:
            raise ValueError(
                f"return column '{self.config.return_column}' not found in predictions frame"
            )

        ordered = frame.sort_values(self.config.timestamp_column)
        timestamps = pd.to_datetime(ordered[self.config.timestamp_column].to_numpy())
        splits = generate_walk_forward_splits(
            timestamps,
            train_window=self.config.train_window,
            test_window=self.config.test_window,
            step=self.config.step,
        )
        results: List[dict[str, object]] = []
        for split in splits:
            test_mask = (
                (timestamps >= split.test_start)
                & (timestamps <= split.test_end)
            )
            test_returns = ordered.loc[test_mask, self.config.return_column].to_numpy(dtype=float)
            if test_returns.size == 0:
                continue
            metrics = self.metrics_fn(test_returns)
            record: dict[str, object] = {
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
            }
            record.update({f"metric_{k}": v for k, v in metrics.items()})
            results.append(record)
        return pd.DataFrame(results)


__all__ = [
    "WalkForwardConfig",
    "WalkForwardSplit",
    "WalkForwardBacktester",
    "generate_walk_forward_splits",
]
