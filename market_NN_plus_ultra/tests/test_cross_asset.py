from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from market_nn_plus_ultra.data.cross_asset import build_cross_asset_view


def _make_price_panel() -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    timeline = pd.date_range(start, periods=6, freq="1D")
    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(["AAA", "BBB", "CCC"]):
        base = 100.0 + idx * 50.0
        close = base + pd.Series(range(len(timeline)), dtype=float)
        volume = (idx + 1) * 1000 + pd.Series(range(len(timeline)), dtype=float) * 10
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timeline,
                    "symbol": symbol,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_build_cross_asset_view_creates_long_frame() -> None:
    price_df = _make_price_panel()
    result = build_cross_asset_view(price_df)

    frame = result.frame
    assert set(frame.columns) == {"timestamp", "feature", "value", "universe", "metadata"}
    assert result.stats.timeline_rows == len(price_df["timestamp"].unique())
    assert "close__AAA" in result.feature_names
    assert "log_return_1__BBB" in result.feature_names
    assert result.stats.fill_rate == pytest.approx(1.0)


def test_build_cross_asset_view_respects_fill_limit() -> None:
    price_df = _make_price_panel()
    # Drop the final observation for one symbol to force a gap that cannot be filled.
    mask = ~(
        (price_df["symbol"] == "BBB")
        & (price_df["timestamp"] == price_df["timestamp"].max())
    )
    trimmed = price_df.loc[mask].reset_index(drop=True)

    result = build_cross_asset_view(
        trimmed,
        value_columns=("close",),
        include_returns=False,
        fill_limit=0,
        forward_fill=False,
        backward_fill=False,
    )

    assert result.stats.fill_rate < 1.0
    assert result.stats.missing_cells > 0
