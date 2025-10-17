"""Market regime labelling utilities for Market NN Plus Ultra."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from ..utils.logging import StructuredLogger, get_structured_logger
from .validation import validate_regime_frame


@dataclass(slots=True)
class VolatilityRegimeConfig:
    """Configuration for realised-volatility regime bucketing."""

    lookback: int = 64
    min_periods: int = 32
    lower_quantile: float = 0.4
    upper_quantile: float = 0.75
    annualisation_factor: float = math.sqrt(252.0)
    unknown_label: str = "volatility_unknown"
    low_label: str = "low_vol"
    mid_label: str = "mid_vol"
    high_label: str = "high_vol"


@dataclass(slots=True)
class LiquidityRegimeConfig:
    """Configuration for liquidity / turnover regime bucketing."""

    lookback: int = 48
    min_periods: int = 16
    lower_quantile: float = 0.3
    upper_quantile: float = 0.7
    smoothing: int = 8
    unknown_label: str = "liquidity_unknown"
    dry_label: str = "dry"
    balanced_label: str = "balanced"
    flood_label: str = "flood"


@dataclass(slots=True)
class RotationRegimeConfig:
    """Configuration for cross-sectional rotation labelling."""

    lookback: int = 96
    lower_quantile: float = 0.25
    upper_quantile: float = 0.75
    min_symbols: int = 2
    price_floor: float = 1e-6
    unknown_label: str = "rotation_unknown"
    laggard_label: str = "sector_laggard"
    neutral_label: str = "sector_neutral"
    leader_label: str = "sector_leader"


@dataclass(slots=True)
class MarketRegimeLabellingConfig:
    """Container bundling all regime labelling configs."""

    volatility: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)
    liquidity: LiquidityRegimeConfig = field(default_factory=LiquidityRegimeConfig)
    rotation: RotationRegimeConfig = field(default_factory=RotationRegimeConfig)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if not np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series, utc=False)
    return series


def _percentile_labels(
    percentiles: pd.Series,
    *,
    lower: float,
    upper: float,
    low_label: str,
    mid_label: str,
    high_label: str,
    unknown_label: str,
) -> pd.Series:
    labels = pd.Series(index=percentiles.index, dtype="object")
    mask = percentiles.notna()
    labels.loc[mask & (percentiles <= lower)] = low_label
    labels.loc[mask & (percentiles >= upper)] = high_label
    labels.loc[mask & labels.isna()] = mid_label
    labels.fillna(unknown_label, inplace=True)
    return labels


def _stable_percentiles(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    order = valid.rank(method="first").astype(float) - 1.0
    denom = max(len(valid) - 1, 1)
    percentiles = order / denom
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[valid.index] = percentiles.to_numpy()
    return result


def _compute_realised_volatility(series: pd.Series, config: VolatilityRegimeConfig) -> pd.Series:
    returns = series.pct_change().fillna(0.0)
    rolling = returns.rolling(window=config.lookback, min_periods=config.min_periods).std()
    vol = rolling * config.annualisation_factor
    return vol


def _volatility_regime(symbol_df: pd.DataFrame, config: VolatilityRegimeConfig) -> pd.Series:
    vol = _compute_realised_volatility(symbol_df["close"], config)
    percentiles = vol.rank(method="average", pct=True)
    return _percentile_labels(
        percentiles,
        lower=config.lower_quantile,
        upper=config.upper_quantile,
        low_label=config.low_label,
        mid_label=config.mid_label,
        high_label=config.high_label,
        unknown_label=config.unknown_label,
    )


def _smoothed_turnover(symbol_df: pd.DataFrame, config: LiquidityRegimeConfig) -> pd.Series:
    if "turnover" in symbol_df.columns:
        base = symbol_df["turnover"].astype(float)
    elif {"close", "volume"}.issubset(symbol_df.columns):
        base = symbol_df["close"].astype(float) * symbol_df["volume"].astype(float)
    else:
        base = pd.Series(np.nan, index=symbol_df.index)
    smoothed = base.rolling(window=config.smoothing, min_periods=1).mean()
    return smoothed.rolling(window=config.lookback, min_periods=config.min_periods).mean()


def _liquidity_regime(symbol_df: pd.DataFrame, config: LiquidityRegimeConfig) -> pd.Series:
    turnover = _smoothed_turnover(symbol_df, config)
    signal = np.log1p(turnover.replace({np.inf: np.nan}))
    percentiles = signal.rank(method="average", pct=True)
    return _percentile_labels(
        percentiles,
        lower=config.lower_quantile,
        upper=config.upper_quantile,
        low_label=config.dry_label,
        mid_label=config.balanced_label,
        high_label=config.flood_label,
        unknown_label=config.unknown_label,
    )


def _rotation_signal(price_df: pd.DataFrame, config: RotationRegimeConfig) -> pd.DataFrame:
    df = price_df.sort_values(["timestamp", "symbol"]).copy()
    grouped = df.groupby("symbol", sort=False)["close"]
    log_prices = grouped.transform(lambda s: np.log(s.clip(lower=config.price_floor)))
    momentum = log_prices - log_prices.shift(config.lookback)
    df["momentum"] = momentum
    df["percentile"] = (
        df.groupby("timestamp")
        ["momentum"]
        .transform(
            lambda s: _stable_percentiles(s) if s.count() >= config.min_symbols else pd.Series(np.nan, index=s.index)
        )
    )
    labels = _percentile_labels(
        df["percentile"],
        lower=config.lower_quantile,
        upper=config.upper_quantile,
        low_label=config.laggard_label,
        mid_label=config.neutral_label,
        high_label=config.leader_label,
        unknown_label=config.unknown_label,
    )
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "symbol": df["symbol"],
            "value": labels,
        }
    )


def _per_symbol_labels(
    symbol: str,
    symbol_df: pd.DataFrame,
    config: MarketRegimeLabellingConfig,
    logger: StructuredLogger,
) -> List[pd.DataFrame]:
    sorted_df = symbol_df.sort_values("timestamp").reset_index(drop=True)
    vol_labels = _volatility_regime(sorted_df, config.volatility)
    liq_labels = _liquidity_regime(sorted_df, config.liquidity)

    vol_counts = vol_labels.value_counts(normalize=True)
    liq_counts = liq_labels.value_counts(normalize=True)

    logger.debug(
        "labelling_summary",
        symbol=symbol,
        volatility_quantiles=(
            float(vol_counts.get(config.volatility.low_label, 0.0)),
            float(vol_counts.get(config.volatility.mid_label, 0.0)),
            float(vol_counts.get(config.volatility.high_label, 0.0)),
        ),
        liquidity_quantiles=(
            float(liq_counts.get(config.liquidity.dry_label, 0.0)),
            float(liq_counts.get(config.liquidity.balanced_label, 0.0)),
            float(liq_counts.get(config.liquidity.flood_label, 0.0)),
        ),
        rows=int(len(sorted_df)),
    )

    frames = [
        pd.DataFrame(
            {
                "timestamp": sorted_df["timestamp"],
                "symbol": symbol,
                "name": "volatility_regime",
                "value": vol_labels.astype(str),
            }
        ),
        pd.DataFrame(
            {
                "timestamp": sorted_df["timestamp"],
                "symbol": symbol,
                "name": "liquidity_regime",
                "value": liq_labels.astype(str),
            }
        ),
    ]
    return frames


def generate_regime_labels(
    price_df: pd.DataFrame,
    *,
    config: MarketRegimeLabellingConfig | None = None,
    assets: pd.DataFrame | None = None,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    """Generate deterministic market regime labels from price/volume data."""

    if price_df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "name", "value"])

    cfg = config or MarketRegimeLabellingConfig()
    structured_logger = logger or get_structured_logger("market_nn_plus_ultra.labelling")

    prepared = price_df.copy()
    prepared["timestamp"] = _ensure_datetime(prepared["timestamp"])
    frames: List[pd.DataFrame] = []

    structured_logger.info(
        "labelling_started",
        symbols=sorted(map(str, prepared["symbol"].astype(str).unique())),
        rows=int(len(prepared)),
    )

    for symbol, group in prepared.groupby(prepared["symbol"].astype(str)):
        frames.extend(_per_symbol_labels(symbol, group, cfg, structured_logger))

    rotation_labels = _rotation_signal(prepared, cfg.rotation)
    rotation_labels["name"] = "rotation_role"
    frames.append(rotation_labels)

    regime_df = pd.concat(frames, ignore_index=True)
    regime_df["timestamp"] = _ensure_datetime(regime_df["timestamp"])
    regime_df["symbol"] = regime_df["symbol"].astype(str)
    regime_df["name"] = regime_df["name"].astype(str)
    regime_df["value"] = regime_df["value"].astype(str)

    validated = validate_regime_frame(regime_df, assets=assets, logger=structured_logger)
    structured_logger.info("labelling_completed", rows=int(len(validated)))
    return validated.reset_index(drop=True)


__all__ = [
    "MarketRegimeLabellingConfig",
    "VolatilityRegimeConfig",
    "LiquidityRegimeConfig",
    "RotationRegimeConfig",
    "generate_regime_labels",
]
