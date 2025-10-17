from __future__ import annotations

import pandas as pd
import pytest

from market_nn_plus_ultra.data.validation import (
    DataValidationError,
    validate_assets_frame,
    validate_indicator_frame,
    validate_price_frame,
    validate_sqlite_frames,
)


def _build_assets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"asset_id": 1, "symbol": "ALPHA", "currency": "USD"},
            {"asset_id": 2, "symbol": "BETA", "currency": "USD"},
        ]
    )


def test_validate_assets_frame_rejects_duplicates() -> None:
    assets = pd.DataFrame(
        [
            {"asset_id": 1, "symbol": "ALPHA"},
            {"asset_id": 2, "symbol": "ALPHA"},
        ]
    )
    with pytest.raises(DataValidationError):
        validate_assets_frame(assets)


def test_validate_price_frame_checks_foreign_keys() -> None:
    assets = validate_assets_frame(_build_assets())
    series = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01 00:00:00"),
                "symbol": "ALPHA",
                "close": 100.0,
            },
            {
                "timestamp": pd.Timestamp("2024-01-01 00:05:00"),
                "symbol": "OMEGA",
                "close": 10.0,
            },
        ]
    )
    with pytest.raises(DataValidationError):
        validate_price_frame(series, assets=assets)


def test_validate_indicator_frame_enforces_uniqueness() -> None:
    assets = validate_assets_frame(_build_assets())
    indicators = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01 00:00:00"),
                "symbol": "ALPHA",
                "name": "rsi_14",
                "value": 50.0,
            },
            {
                "timestamp": pd.Timestamp("2024-01-01 00:00:00"),
                "symbol": "ALPHA",
                "name": "rsi_14",
                "value": 51.0,
            },
        ]
    )
    with pytest.raises(DataValidationError):
        validate_indicator_frame(indicators, assets=assets)


def test_validate_sqlite_frames_returns_bundle() -> None:
    assets = validate_assets_frame(_build_assets())
    series = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01 00:00:00"),
                "symbol": "ALPHA",
                "close": 100.0,
            },
            {
                "timestamp": pd.Timestamp("2024-01-01 00:05:00"),
                "symbol": "BETA",
                "close": 98.0,
            },
        ]
    )
    indicators = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01 00:00:00"),
                "symbol": "ALPHA",
                "name": "rsi_14",
                "value": 42.0,
            }
        ]
    )

    bundle = validate_sqlite_frames(
        {"assets": assets, "series": series, "indicators": indicators}
    )

    assert bundle.assets is not None
    assert bundle.series is not None
    assert bundle.indicators is not None
