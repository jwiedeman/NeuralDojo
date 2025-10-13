"""Schema validation helpers for Market NN Plus Ultra datasets."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pandera as pa


PRICE_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(pa.DateTime, nullable=False, coerce=True),
        "symbol": pa.Column(pa.String, nullable=False, coerce=True),
        "open": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
        "high": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
        "low": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
        "close": pa.Column(pa.Float, nullable=False, coerce=True),
        "volume": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
        "vwap": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
        "turnover": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
    },
    coerce=True,
)


INDICATOR_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(pa.DateTime, nullable=False, coerce=True),
        "symbol": pa.Column(pa.String, nullable=False, coerce=True),
        "name": pa.Column(pa.String, nullable=True, required=False, coerce=True),
        "value": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
    },
    coerce=True,
)


def _ensure_unique(df: pd.DataFrame, subset: Iterable[str], context: str) -> None:
    duplicates = df[df.duplicated(subset=list(subset), keep=False)]
    if not duplicates.empty:
        sample = duplicates[list(subset)].head(5).to_dict("records")
        raise ValueError(
            f"{context} contains duplicate rows for keys {list(subset)}. "
            f"First duplicates: {sample}"
        )


def _ensure_sorted(df: pd.DataFrame, context: str) -> None:
    for symbol, group in df.groupby("symbol"):
        if not group["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"{context} for symbol '{symbol}' is not sorted by timestamp."
            )


def validate_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV rows before feature engineering."""

    validated = PRICE_SCHEMA.validate(df, lazy=True)
    _ensure_unique(validated, ("timestamp", "symbol"), context="Price series")
    _ensure_sorted(validated.sort_values(["symbol", "timestamp"]), context="Price series")
    return validated


def validate_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate indicator rows that will be merged with the price series."""

    if df.empty:
        return df
    validated = INDICATOR_SCHEMA.validate(df, lazy=True)
    subset = ["timestamp", "symbol"]
    if "name" in validated.columns:
        subset.append("name")
    _ensure_unique(validated, subset, context="Indicator table")
    return validated


def safe_float_array(series: pd.Series) -> np.ndarray:
    """Return a float32 numpy array with NaNs replaced by zeros."""

    values = series.to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(values, copy=False)
    return values


__all__ = [
    "validate_price_frame",
    "validate_indicator_frame",
    "safe_float_array",
]

