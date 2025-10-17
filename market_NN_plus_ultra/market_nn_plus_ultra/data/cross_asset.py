"""Utilities for constructing cross-asset feature views."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import StructuredLogger, get_structured_logger


@dataclass(slots=True)
class CrossAssetViewStats:
    """Summary statistics describing a generated cross-asset view."""

    timeline_rows: int
    feature_columns: int
    dropped_rows: int
    dropped_features: tuple[str, ...]
    missing_cells: int
    total_cells: int
    universe: tuple[str, ...]

    @property
    def fill_rate(self) -> float:
        """Return the proportion of populated cells after alignment."""

        if self.total_cells == 0:
            return 0.0
        return 1.0 - (self.missing_cells / self.total_cells)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the stats."""

        return {
            "timeline_rows": self.timeline_rows,
            "feature_columns": self.feature_columns,
            "dropped_rows": self.dropped_rows,
            "dropped_features": list(self.dropped_features),
            "missing_cells": self.missing_cells,
            "total_cells": self.total_cells,
            "fill_rate": self.fill_rate,
            "universe": list(self.universe),
        }


@dataclass(slots=True)
class CrossAssetViewResult:
    """Container for the cross-asset view frame and metadata."""

    frame: pd.DataFrame
    stats: CrossAssetViewStats
    feature_names: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable payload describing the result."""

        payload = self.stats.to_dict()
        payload.update({
            "rows": len(self.frame),
            "feature_names": self.feature_names,
        })
        return payload


def _normalise_logger(logger: logging.Logger | StructuredLogger | None) -> StructuredLogger:
    if isinstance(logger, StructuredLogger):
        return logger
    if isinstance(logger, logging.Logger):
        return StructuredLogger(logger)
    return get_structured_logger("market_nn_plus_ultra.cross_asset")


def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=False)
    prepared["symbol"] = prepared["symbol"].astype(str)
    return prepared.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "price dataframe is missing required columns: " f"{', '.join(sorted(missing))}"
        )


def _feature_metadata(feature: str) -> str:
    if "__" in feature:
        field, symbol = feature.split("__", 1)
    else:
        field, symbol = feature, ""
    payload = {"field": field, "symbol": symbol}
    return json.dumps(payload, sort_keys=True)


def build_cross_asset_view(
    price_df: pd.DataFrame,
    *,
    value_columns: Sequence[str] = ("close", "volume"),
    include_returns: bool = True,
    fill_limit: int | None = None,
    forward_fill: bool = True,
    backward_fill: bool = True,
    universe_name: str | None = None,
    logger: logging.Logger | StructuredLogger | None = None,
) -> CrossAssetViewResult:
    """Return an aligned cross-asset view for the provided price panel."""

    if price_df.empty:
        raise ValueError("price dataframe is empty; cannot build cross-asset view")

    working = _prepare_price_frame(price_df)
    symbols = tuple(sorted(working["symbol"].unique()))
    if not symbols:
        raise ValueError("price dataframe does not contain any symbols")

    fields: list[str] = list(dict.fromkeys(value_columns))
    required_columns: set[str] = set(fields)
    if include_returns:
        required_columns.add("close")
    _ensure_columns(working, required_columns)

    if include_returns:
        grouped = working.groupby("symbol", group_keys=False)["close"]
        log_returns = grouped.transform(
            lambda series: np.log(series.astype(float)).diff().fillna(0.0)
        )
        working["log_return_1"] = log_returns.astype(float)
        fields = list(dict.fromkeys([*fields, "log_return_1"]))

    timeline = pd.DatetimeIndex(sorted(working["timestamp"].unique()))
    structured_logger = _normalise_logger(logger)

    wide_frames: list[pd.DataFrame] = []
    dropped_features: list[str] = []
    for field in fields:
        pivot = working.pivot(index="timestamp", columns="symbol", values=field).sort_index()
        pivot = pivot.reindex(timeline)
        if forward_fill:
            pivot = pivot.ffill(limit=fill_limit)
        if backward_fill:
            pivot = pivot.bfill(limit=fill_limit)
        pivot.columns = [f"{field}__{symbol}" for symbol in pivot.columns]
        wide_frames.append(pivot)

    wide = pd.concat(wide_frames, axis=1)
    if wide.empty:
        raise ValueError("cross-asset alignment produced no features")

    valid_columns = wide.notna().any(axis=0)
    if not valid_columns.all():
        dropped_features = [column for column, keep in valid_columns.items() if not keep]
        wide = wide.loc[:, valid_columns]

    valid_rows = wide.notna().any(axis=1)
    dropped_rows = int((~valid_rows).sum())
    wide = wide.loc[valid_rows]

    total_cells = int(wide.shape[0] * wide.shape[1])
    missing_cells = int(wide.isna().sum().sum())
    feature_names = list(wide.columns)

    long = (
        wide.sort_index()
        .stack(future_stack=True)
        .rename_axis(index=["timestamp", "feature"])
        .reset_index(name="value")
    )
    long = long.dropna(subset=["value"])

    universe = universe_name or ",".join(symbols)
    long["universe"] = universe
    metadata_map = {feature: _feature_metadata(feature) for feature in feature_names}
    long["metadata"] = long["feature"].map(metadata_map)

    stats = CrossAssetViewStats(
        timeline_rows=int(wide.shape[0]),
        feature_columns=len(feature_names),
        dropped_rows=dropped_rows,
        dropped_features=tuple(dropped_features),
        missing_cells=missing_cells,
        total_cells=total_cells,
        universe=symbols,
    )

    structured_logger.info(
        "cross_asset_view_built",
        rows=len(long),
        fill_rate=stats.fill_rate,
        dropped_rows=dropped_rows,
        dropped_features=list(dropped_features),
        features=feature_names,
        universe=list(symbols),
    )

    return CrossAssetViewResult(frame=long, stats=stats, feature_names=feature_names)


__all__ = ["CrossAssetViewResult", "CrossAssetViewStats", "build_cross_asset_view"]

