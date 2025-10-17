"""Tests for the sliding window dataset utilities."""

import warnings

import numpy as np
import pandas as pd
import pytest

from market_nn_plus_ultra.data.window_dataset import SlidingWindowDataset


def _build_panel() -> pd.DataFrame:
    timestamps = pd.date_range("2021-01-01", periods=12, freq="h")
    index = pd.MultiIndex.from_product([timestamps, ["XYZ"]], names=["timestamp", "symbol"])
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, len(index)),
            "feature_b": np.linspace(1.0, 2.0, len(index)),
            "close": np.linspace(100.0, 110.0, len(index)),
        },
        index=index,
    )
    # Introduce problematic values typical of indicator warm-up periods.
    frame.loc[(timestamps[0], "XYZ"), "feature_a"] = np.nan
    frame.loc[(timestamps[1], "XYZ"), "feature_b"] = np.inf
    frame.loc[(timestamps[2], "XYZ"), "feature_a"] = -np.inf
    return frame


def test_dataset_returns_finite_tensors() -> None:
    panel = _build_panel()
    dataset = SlidingWindowDataset(
        panel,
        feature_columns=["feature_a", "feature_b"],
        target_columns=["close"],
        window_size=4,
        horizon=2,
        stride=1,
        normalise=True,
    )

    sample = dataset[0]

    assert sample["features"].isfinite().all(), "features should not contain NaNs or infinities"
    assert sample["targets"].isfinite().all(), "targets should not contain NaNs or infinities"
    assert sample["reference"].isfinite().all(), "reference should not contain NaNs or infinities"


def test_normalisation_handles_all_nan_feature_column() -> None:
    panel = _build_panel()
    panel["feature_a"] = np.nan
    dataset = SlidingWindowDataset(
        panel,
        feature_columns=["feature_a", "feature_b"],
        target_columns=["close"],
        window_size=4,
        horizon=2,
        stride=1,
        normalise=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample = dataset[0]

    assert not caught, "normalisation should not emit warnings for all-NaN columns"
    feature_a = sample["features"][:, 0]
    assert feature_a.isfinite().all()
    assert feature_a.abs().max().item() == pytest.approx(0.0)
