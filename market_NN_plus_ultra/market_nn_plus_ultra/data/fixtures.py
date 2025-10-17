"""Synthetic fixture generation utilities for Market NN Plus Ultra."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .labelling import generate_regime_labels
from .validation import (
    validate_assets_frame,
    validate_indicator_frame,
    validate_price_frame,
    validate_regime_frame,
    validate_sqlite_frames,
)


@dataclass(slots=True)
class FixtureConfig:
    """Configuration values controlling fixture generation."""

    symbols: list[str]
    rows: int
    freq: str
    seed: int
    start: datetime
    alt_features: int

    @property
    def timeline(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.rows, freq=self.freq)


def _rolling_feature(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _volatility_feature(series: pd.Series, window: int) -> pd.Series:
    returns = series.pct_change().fillna(0.0)
    return returns.rolling(window=window, min_periods=1).std().fillna(0.0)


def _generate_price_panel(config: FixtureConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    frames: list[pd.DataFrame] = []

    for symbol in config.symbols:
        sym_seed = rng.integers(0, 2**31 - 1)
        sym_rng = np.random.default_rng(sym_seed)
        base_price = sym_rng.uniform(40, 400)
        drift = sym_rng.normal(0.0002, 0.00005)
        volatility = sym_rng.uniform(0.01, 0.04)
        returns = sym_rng.normal(drift, volatility, size=config.rows)
        close = base_price * np.exp(np.cumsum(returns))
        open_price = close * (1 + sym_rng.normal(0, volatility * 0.5, size=config.rows))
        high = np.maximum(open_price, close) * (1 + sym_rng.uniform(0.0001, 0.01, size=config.rows))
        low = np.minimum(open_price, close) * (1 - sym_rng.uniform(0.0001, 0.01, size=config.rows))
        volume = sym_rng.lognormal(mean=14, sigma=0.25, size=config.rows)
        vwap = (high + low + close) / 3
        turnover = volume * close

        df = pd.DataFrame(
            {
                "timestamp": config.timeline,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "vwap": vwap,
                "turnover": turnover,
            }
        )
        frames.append(df)

    price_df = pd.concat(frames, ignore_index=True)
    price_df = validate_price_frame(price_df)
    return price_df


def _generate_indicator_table(price_df: pd.DataFrame, config: FixtureConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for symbol, sym_df in price_df.groupby("symbol"):
        sym_df = sym_df.sort_values("timestamp").reset_index(drop=True)
        indicators = pd.DataFrame({"timestamp": sym_df["timestamp"], "symbol": symbol})
        indicators["ma_24"] = _rolling_feature(sym_df["close"], 24)
        indicators["ma_96"] = _rolling_feature(sym_df["close"], 96)
        indicators["vol_96"] = _volatility_feature(sym_df["close"], 96)
        indicators["momentum_32"] = sym_df["close"].pct_change(periods=32).fillna(0.0)
        indicators["drawdown"] = (sym_df["close"] / sym_df["close"].cummax()) - 1.0

        alt_cols = {}
        for idx in range(config.alt_features):
            signal = sym_df["close"].pct_change(periods=idx + 1).rolling(window=24, min_periods=1).mean()
            noise_rng = np.random.default_rng(config.seed + idx)
            alt_cols[f"alt_signal_{idx + 1}"] = signal.fillna(0.0) + noise_rng.normal(scale=0.01, size=signal.size)
        if alt_cols:
            indicators = indicators.join(pd.DataFrame(alt_cols))

        melted = indicators.melt(id_vars=["timestamp", "symbol"], var_name="name", value_name="value")
        frames.append(melted)

    indicator_df = pd.concat(frames, ignore_index=True)
    indicator_df = validate_indicator_frame(indicator_df)
    return indicator_df


def _generate_assets(symbols: Iterable[str]) -> pd.DataFrame:
    data = [
        {
            "asset_id": idx + 1,
            "symbol": symbol,
            "sector": "simulated",
            "currency": "USD",
            "exchange": "SIM",
            "metadata": json.dumps({"source": "synthetic_fixture"}),
        }
        for idx, symbol in enumerate(symbols)
    ]
    assets = pd.DataFrame(data)
    return validate_assets_frame(assets)


def _generate_regime_table(price_df: pd.DataFrame, assets: pd.DataFrame | None = None) -> pd.DataFrame:
    return generate_regime_labels(price_df, assets=assets)


def build_fixture(config: FixtureConfig) -> dict[str, pd.DataFrame]:
    """Return the generated fixture tables."""

    price_df = _generate_price_panel(config)
    assets_df = _generate_assets(config.symbols)
    price_df = validate_price_frame(price_df, assets=assets_df)
    indicator_df = _generate_indicator_table(price_df, config)
    indicator_df = validate_indicator_frame(indicator_df, assets=assets_df)
    regime_df = _generate_regime_table(price_df, assets=assets_df)
    regime_df = validate_regime_frame(regime_df, assets=assets_df)
    frames = {
        "series": price_df,
        "indicators": indicator_df,
        "regimes": regime_df,
        "assets": assets_df,
    }
    validate_sqlite_frames(frames)  # sanity check bundle relationships
    return frames


def write_fixture(frames: dict[str, pd.DataFrame], db_path: Path) -> Path:
    """Persist the generated tables into a SQLite database."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        frames["series"].to_sql("series", conn, index=False, if_exists="replace")
        frames["indicators"].to_sql("indicators", conn, index=False, if_exists="replace")
        frames["regimes"].to_sql("regimes", conn, index=False, if_exists="replace")
        frames["assets"].to_sql("assets", conn, index=False, if_exists="replace")
    return db_path


__all__ = ["FixtureConfig", "build_fixture", "write_fixture"]
