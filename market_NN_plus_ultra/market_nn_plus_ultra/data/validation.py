"""Schema validation helpers for Market NN Plus Ultra datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import StructuredLogger, get_structured_logger
from ..reporting.annotations import DECISION_CHOICES

try:  # pragma: no cover - optional dependency fallback
    import pandera.errors as pa_errors  # type: ignore
    import pandera.pandas as pa  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    pa = None
    pa_errors = None


class DataValidationError(ValueError):
    """Raised when a dataframe fails schema or integrity validation."""

    def __init__(self, context: str, message: str) -> None:
        super().__init__(f"{context}: {message}")
        self.context = context
        self.details = message


def _get_logger(logger: StructuredLogger | None) -> StructuredLogger:
    if logger is not None:
        return logger
    return get_structured_logger("market_nn_plus_ultra.validation")


def _log_failure(logger: StructuredLogger | None, context: str, *, errors: Mapping[str, object] | Sequence[Mapping[str, object]] | None, reason: str) -> None:
    structured = _get_logger(logger)
    payload: MutableMapping[str, object] = {"context": context, "reason": reason}
    if errors is not None:
        payload["errors"] = errors
    structured.error("data_validation_failed", **payload)


def _ensure_unique(df: pd.DataFrame, subset: Iterable[str], context: str, *, logger: StructuredLogger | None = None) -> None:
    duplicates = df[df.duplicated(subset=list(subset), keep=False)]
    if not duplicates.empty:
        sample = duplicates[list(subset)].head(5).to_dict("records")
        _log_failure(logger, context, errors=sample, reason="duplicate_rows")
        raise DataValidationError(context, f"duplicate rows for keys {list(subset)}")


def _ensure_sorted(df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> None:
    if "timestamp" not in df.columns or "symbol" not in df.columns:
        return
    for symbol, group in df.groupby("symbol"):
        if not group["timestamp"].is_monotonic_increasing:
            _log_failure(
                logger,
                context,
                errors={"symbol": symbol, "timestamps": group["timestamp"].head(5).tolist()},
                reason="unsorted_timestamps",
            )
            raise DataValidationError(context, f"timestamps for '{symbol}' must be sorted")


def _run_pandera_validation(schema: pa.DataFrameSchema, df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> pd.DataFrame:  # pragma: no cover - delegated to Pandera
    try:
        return schema.validate(df, lazy=True)
    except pa_errors.SchemaErrors as exc:  # type: ignore[attr-defined]
        errors = exc.failure_cases.to_dict("records")
        _log_failure(logger, context, errors=errors, reason="schema_mismatch")
        raise DataValidationError(context, "failed pandera schema validation") from exc


def _coerce_datetime(series: pd.Series, context: str, *, logger: StructuredLogger | None = None) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="raise", utc=False)
    except Exception as exc:  # pragma: no cover - extremely rare branch
        _log_failure(logger, context, errors={"column": series.name, "sample": series.head().tolist()}, reason="datetime_parse_error")
        raise DataValidationError(context, f"failed to parse datetimes for column '{series.name}'") from exc


def _fallback_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fallback_price_validation(df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    required = {"timestamp", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        _log_failure(logger, context, errors={"missing_columns": sorted(missing)}, reason="missing_columns")
        raise DataValidationError(context, f"missing columns: {sorted(missing)}")
    validated = df.copy()
    validated["timestamp"] = _coerce_datetime(validated["timestamp"], context, logger=logger)
    validated["symbol"] = validated["symbol"].astype(str)
    numeric_cols = [
        col
        for col in ("open", "high", "low", "close", "volume", "vwap", "turnover")
        if col in validated.columns
    ]
    for col in numeric_cols:
        validated[col] = _fallback_numeric(validated[col])
    return validated


def _fallback_indicator_validation(df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    validated = df.copy()
    validated["timestamp"] = _coerce_datetime(validated["timestamp"], context, logger=logger)
    validated["symbol"] = validated["symbol"].astype(str)
    if "value" in validated.columns:
        validated["value"] = _fallback_numeric(validated["value"])
    if "name" in validated.columns:
        validated["name"] = validated["name"].astype(str)
    if "metadata" in validated.columns:
        validated["metadata"] = validated["metadata"].astype(str)
    return validated


def _fallback_assets_validation(df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    required = {"symbol"}
    missing = required - set(df.columns)
    if missing:
        _log_failure(logger, context, errors={"missing_columns": sorted(missing)}, reason="missing_columns")
        raise DataValidationError(context, f"missing columns: {sorted(missing)}")
    validated = df.copy()
    if "asset_id" in validated.columns:
        validated["asset_id"] = pd.to_numeric(validated["asset_id"], errors="coerce").astype("Int64")
    validated["symbol"] = validated["symbol"].astype(str)
    for col in ("sector", "currency", "exchange", "metadata"):
        if col in validated.columns:
            validated[col] = validated[col].astype(str)
    return validated


def _fallback_regime_validation(df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    validated = df.copy()
    validated["timestamp"] = _coerce_datetime(validated["timestamp"], context, logger=logger)
    validated["symbol"] = validated["symbol"].astype(str)
    if "name" in validated.columns:
        validated["name"] = validated["name"].astype(str)
    if "value" in validated.columns:
        validated["value"] = validated["value"].astype(str)
    return validated


def _fallback_trades_validation(df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    validated = df.copy()
    if "trade_id" in validated.columns:
        validated["trade_id"] = pd.to_numeric(validated["trade_id"], errors="coerce").astype("Int64")
    if "timestamp" in validated.columns:
        validated["timestamp"] = _coerce_datetime(validated["timestamp"], context, logger=logger)
    if "symbol" in validated.columns:
        validated["symbol"] = validated["symbol"].astype(str)
    for col in ("side",):
        if col in validated.columns:
            validated[col] = validated[col].astype(str)
    for col in ("size", "price", "fees", "slippage_bp", "pnl"):
        if col in validated.columns:
            validated[col] = _fallback_numeric(validated[col])
    if "metadata" in validated.columns:
        validated["metadata"] = validated["metadata"].astype(str)
    return validated


def _fallback_benchmark_validation(df: pd.DataFrame, context: str, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    validated = df.copy()
    validated["timestamp"] = _coerce_datetime(validated["timestamp"], context, logger=logger)
    validated["symbol"] = validated["symbol"].astype(str)
    for col in ("return", "level"):
        if col in validated.columns:
            validated[col] = _fallback_numeric(validated[col])
    return validated


def _fallback_cross_asset_validation(
    df: pd.DataFrame,
    context: str,
    *,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    validated = df.copy()
    validated["timestamp"] = _coerce_datetime(validated["timestamp"], context, logger=logger)
    validated["feature"] = validated["feature"].astype(str)
    validated["value"] = _fallback_numeric(validated["value"])
    if "universe" in validated.columns:
        validated["universe"] = validated["universe"].astype(str)
    if "metadata" in validated.columns:
        validated["metadata"] = validated["metadata"].astype(str)
    return validated


def _fallback_annotation_validation(
    df: pd.DataFrame,
    context: str,
    *,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    validated = df.copy()
    if "annotation_id" in validated.columns:
        validated["annotation_id"] = pd.to_numeric(validated["annotation_id"], errors="coerce").astype("Int64")
    if "trade_id" in validated.columns:
        validated["trade_id"] = pd.to_numeric(validated["trade_id"], errors="coerce").astype("Int64")
    for column in ("trade_timestamp", "created_at", "context_window_start", "context_window_end"):
        if column in validated.columns:
            validated[column] = _coerce_datetime(validated[column], context, logger=logger)
    for column in ("symbol", "decision", "rationale", "author", "tags"):
        if column in validated.columns:
            validated[column] = validated[column].astype(str)
    if "confidence" in validated.columns:
        validated["confidence"] = _fallback_numeric(validated["confidence"])
    if "metadata" in validated.columns:
        validated["metadata"] = validated["metadata"].astype(str)
    return validated


if pa is not None:  # pragma: no branch - only executed when dependency available
    ASSET_SCHEMA = pa.DataFrameSchema(
        {
            "asset_id": pa.Column(pa.Int, nullable=True, required=False, coerce=True),
            "symbol": pa.Column(pa.String, nullable=False, coerce=True),
            "sector": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "currency": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "exchange": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "metadata": pa.Column(pa.String, nullable=True, required=False, coerce=True),
        },
        coerce=True,
    )

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
            "metadata": pa.Column(pa.String, nullable=True, required=False, coerce=True),
        },
        coerce=True,
    )

    REGIME_SCHEMA = pa.DataFrameSchema(
        {
            "timestamp": pa.Column(pa.DateTime, nullable=False, coerce=True),
            "symbol": pa.Column(pa.String, nullable=False, coerce=True),
            "name": pa.Column(pa.String, nullable=False, coerce=True),
            "value": pa.Column(pa.String, nullable=False, coerce=True),
        },
        coerce=True,
    )

    TRADE_SCHEMA = pa.DataFrameSchema(
        {
            "trade_id": pa.Column(pa.Int, nullable=True, required=False, coerce=True),
            "timestamp": pa.Column(pa.DateTime, nullable=False, coerce=True),
            "symbol": pa.Column(pa.String, nullable=False, coerce=True),
            "side": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "size": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
            "price": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
            "fees": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
            "slippage_bp": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
            "pnl": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
            "metadata": pa.Column(pa.String, nullable=True, required=False, coerce=True),
        },
        coerce=True,
    )

    BENCHMARK_SCHEMA = pa.DataFrameSchema(
        {
            "timestamp": pa.Column(pa.DateTime, nullable=False, coerce=True),
            "symbol": pa.Column(pa.String, nullable=False, coerce=True),
            "return": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
            "level": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
        },
        coerce=True,
    )

    CROSS_ASSET_VIEW_SCHEMA = pa.DataFrameSchema(
        {
            "timestamp": pa.Column(pa.DateTime, nullable=False, coerce=True),
            "feature": pa.Column(pa.String, nullable=False, coerce=True),
            "value": pa.Column(pa.Float, nullable=False, coerce=True),
            "universe": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "metadata": pa.Column(pa.String, nullable=True, required=False, coerce=True),
        },
        coerce=True,
    )

    TRADE_ANNOTATION_SCHEMA = pa.DataFrameSchema(
        {
            "annotation_id": pa.Column(pa.Int, nullable=True, required=False, coerce=True),
            "trade_id": pa.Column(pa.Int, nullable=False, coerce=True),
            "symbol": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "trade_timestamp": pa.Column(pa.DateTime, nullable=True, required=False, coerce=True),
            "decision": pa.Column(
                pa.String,
                nullable=False,
                coerce=True,
                checks=pa.Check.isin(list(DECISION_CHOICES)),
            ),
            "rationale": pa.Column(pa.String, nullable=False, coerce=True),
            "confidence": pa.Column(pa.Float, nullable=True, required=False, coerce=True),
            "tags": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "author": pa.Column(pa.String, nullable=False, coerce=True),
            "created_at": pa.Column(pa.DateTime, nullable=False, coerce=True),
            "metadata": pa.Column(pa.String, nullable=True, required=False, coerce=True),
            "context_window_start": pa.Column(pa.DateTime, nullable=True, required=False, coerce=True),
            "context_window_end": pa.Column(pa.DateTime, nullable=True, required=False, coerce=True),
        },
        coerce=True,
    )
else:  # pragma: no cover - fallback definitions
    ASSET_SCHEMA = None
    PRICE_SCHEMA = None
    INDICATOR_SCHEMA = None
    REGIME_SCHEMA = None
    TRADE_SCHEMA = None
    BENCHMARK_SCHEMA = None
    CROSS_ASSET_VIEW_SCHEMA = None
    TRADE_ANNOTATION_SCHEMA = None


def _enforce_foreign_key(df: pd.DataFrame, column: str, valid_values: set[str], context: str, *, logger: StructuredLogger | None = None) -> None:
    if column not in df.columns:
        return
    missing = sorted(str(value) for value in set(df[column].astype(str)) - valid_values)
    if missing:
        _log_failure(
            logger,
            context,
            errors={"column": column, "unknown_values": missing[:5]},
            reason="foreign_key_violation",
        )
        raise DataValidationError(context, f"contains values in '{column}' with no matching asset")


def validate_assets_frame(df: pd.DataFrame, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    """Validate the assets dimension table."""

    context = "assets_table"
    if df.empty:
        _log_failure(logger, context, errors=None, reason="empty_frame")
        raise DataValidationError(context, "no assets supplied")
    if ASSET_SCHEMA is not None:
        validated = _run_pandera_validation(ASSET_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_assets_validation(df, context, logger=logger)
    _ensure_unique(validated, ("symbol",), context, logger=logger)
    return validated


def validate_price_frame(df: pd.DataFrame, *, assets: pd.DataFrame | None = None, logger: StructuredLogger | None = None) -> pd.DataFrame:
    """Validate OHLCV series before feature engineering."""

    context = "series_table"
    if PRICE_SCHEMA is not None:
        validated = _run_pandera_validation(PRICE_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_price_validation(df, context, logger=logger)
    validated = validated.sort_values(["symbol", "timestamp"])
    _ensure_unique(validated, ("timestamp", "symbol"), context, logger=logger)
    _ensure_sorted(validated, context, logger=logger)
    if assets is not None and not assets.empty:
        valid_symbols = set(assets["symbol"].astype(str))
        _enforce_foreign_key(validated, "symbol", valid_symbols, context, logger=logger)
    return validated


def validate_indicator_frame(
    df: pd.DataFrame,
    *,
    assets: pd.DataFrame | None = None,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    """Validate indicator rows that will be merged with the price series."""

    if df.empty:
        return df
    context = "indicators_table"
    if INDICATOR_SCHEMA is not None:
        validated = _run_pandera_validation(INDICATOR_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_indicator_validation(df, context, logger=logger)
    subset = ["timestamp", "symbol"]
    if "name" in validated.columns:
        subset.append("name")
    _ensure_unique(validated, subset, context, logger=logger)
    if assets is not None and not assets.empty:
        valid_symbols = set(assets["symbol"].astype(str))
        _enforce_foreign_key(validated, "symbol", valid_symbols, context, logger=logger)
    return validated.sort_values(["symbol", "timestamp"])


def validate_regime_frame(
    df: pd.DataFrame,
    *,
    assets: pd.DataFrame | None = None,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    """Validate market regime labelling tables."""

    if df.empty:
        return df
    context = "regimes_table"
    if REGIME_SCHEMA is not None:
        validated = _run_pandera_validation(REGIME_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_regime_validation(df, context, logger=logger)
    subset = ["timestamp", "symbol"]
    if "name" in validated.columns:
        subset.append("name")
    _ensure_unique(validated, subset, context, logger=logger)
    if assets is not None and not assets.empty:
        valid_symbols = set(assets["symbol"].astype(str))
        _enforce_foreign_key(validated, "symbol", valid_symbols, context, logger=logger)
    return validated.sort_values(["symbol", "timestamp"])


def validate_trades_frame(
    df: pd.DataFrame,
    *,
    assets: pd.DataFrame | None = None,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    """Validate trade execution logs before evaluation."""

    if df.empty:
        return df
    context = "trades_table"
    if TRADE_SCHEMA is not None:
        validated = _run_pandera_validation(TRADE_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_trades_validation(df, context, logger=logger)
    key = "trade_id" if "trade_id" in validated.columns else None
    if key:
        _ensure_unique(validated, (key,), context, logger=logger)
    if assets is not None and not assets.empty:
        valid_symbols = set(assets["symbol"].astype(str))
        _enforce_foreign_key(validated, "symbol", valid_symbols, context, logger=logger)
    return validated.sort_values([col for col in ("symbol", "timestamp") if col in validated.columns])


def validate_trade_annotation_frame(
    df: pd.DataFrame,
    *,
    trades: pd.DataFrame | None = None,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    """Validate analyst feedback annotations."""

    if df.empty:
        return df
    context = "trade_annotations_table"
    if TRADE_ANNOTATION_SCHEMA is not None:
        validated = _run_pandera_validation(TRADE_ANNOTATION_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_annotation_validation(df, context, logger=logger)
    subset = ["trade_id", "created_at"]
    if "annotation_id" in validated.columns:
        subset.append("annotation_id")
    _ensure_unique(validated, subset, context, logger=logger)
    if trades is not None and not trades.empty and "trade_id" in trades.columns:
        valid_ids = set(trades["trade_id"].dropna().astype(int))
        supplied = set(validated["trade_id"].dropna().astype(int))
        missing = sorted(supplied - valid_ids)
        if missing:
            _log_failure(
                logger,
                context,
                errors={"unknown_trade_ids": missing[:5]},
                reason="foreign_key_violation",
            )
            raise DataValidationError(context, "contains references to trades not present in the dataset")
    return validated.sort_values("created_at", ascending=False)


def validate_benchmark_frame(df: pd.DataFrame, *, logger: StructuredLogger | None = None) -> pd.DataFrame:
    """Validate benchmark return series."""

    if df.empty:
        return df
    context = "benchmarks_table"
    if BENCHMARK_SCHEMA is not None:
        validated = _run_pandera_validation(BENCHMARK_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_benchmark_validation(df, context, logger=logger)
    _ensure_unique(validated, ("timestamp", "symbol"), context, logger=logger)
    return validated.sort_values(["symbol", "timestamp"])


def validate_cross_asset_view_frame(
    df: pd.DataFrame,
    *,
    logger: StructuredLogger | None = None,
) -> pd.DataFrame:
    """Validate aligned cross-asset feature views."""

    if df.empty:
        return df

    context = "cross_asset_views"
    if CROSS_ASSET_VIEW_SCHEMA is not None:
        validated = _run_pandera_validation(CROSS_ASSET_VIEW_SCHEMA, df, context, logger=logger)
    else:
        validated = _fallback_cross_asset_validation(df, context, logger=logger)

    _ensure_unique(validated, ("timestamp", "feature"), context, logger=logger)
    return validated.sort_values(["timestamp", "feature"]).reset_index(drop=True)


@dataclass(slots=True)
class ValidationBundle:
    """Container for validated frames returned by :func:`validate_sqlite_frames`."""

    assets: pd.DataFrame | None = None
    series: pd.DataFrame | None = None
    indicators: pd.DataFrame | None = None
    regimes: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    benchmarks: pd.DataFrame | None = None
    cross_asset_views: pd.DataFrame | None = None
    trade_annotations: pd.DataFrame | None = None


def validate_sqlite_frames(frames: Mapping[str, pd.DataFrame], *, logger: StructuredLogger | None = None) -> ValidationBundle:
    """Validate a dictionary of SQLite tables and return the validated bundle."""

    assets = frames.get("assets")
    validated_assets = validate_assets_frame(assets, logger=logger) if assets is not None else None

    series = frames.get("series")
    validated_series = (
        validate_price_frame(series, assets=validated_assets, logger=logger) if series is not None else None
    )

    indicators = frames.get("indicators")
    validated_indicators = (
        validate_indicator_frame(indicators, assets=validated_assets, logger=logger)
        if indicators is not None
        else None
    )

    regimes = frames.get("regimes")
    validated_regimes = (
        validate_regime_frame(regimes, assets=validated_assets, logger=logger) if regimes is not None else None
    )

    trades = frames.get("trades")
    validated_trades = (
        validate_trades_frame(trades, assets=validated_assets, logger=logger) if trades is not None else None
    )

    annotations = frames.get("trade_annotations")
    validated_annotations = (
        validate_trade_annotation_frame(annotations, trades=validated_trades, logger=logger)
        if annotations is not None
        else None
    )

    benchmarks = frames.get("benchmarks")
    validated_benchmarks = (
        validate_benchmark_frame(benchmarks, logger=logger) if benchmarks is not None else None
    )

    cross_asset = frames.get("cross_asset_views")
    validated_cross_asset = (
        validate_cross_asset_view_frame(cross_asset, logger=logger)
        if cross_asset is not None
        else None
    )

    return ValidationBundle(
        assets=validated_assets,
        series=validated_series,
        indicators=validated_indicators,
        regimes=validated_regimes,
        trades=validated_trades,
        benchmarks=validated_benchmarks,
        cross_asset_views=validated_cross_asset,
        trade_annotations=validated_annotations,
    )


def safe_float_array(series: pd.Series) -> np.ndarray:
    """Return a float32 numpy array with NaNs replaced by zeros."""

    values = series.to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(values, copy=False)
    return values


__all__ = [
    "ASSET_SCHEMA",
    "BENCHMARK_SCHEMA",
    "CROSS_ASSET_VIEW_SCHEMA",
    "DataValidationError",
    "INDICATOR_SCHEMA",
    "PRICE_SCHEMA",
    "REGIME_SCHEMA",
    "TRADE_SCHEMA",
    "TRADE_ANNOTATION_SCHEMA",
    "ValidationBundle",
    "validate_assets_frame",
    "validate_benchmark_frame",
    "validate_cross_asset_view_frame",
    "validate_indicator_frame",
    "validate_price_frame",
    "validate_regime_frame",
    "validate_sqlite_frames",
    "validate_trade_annotation_frame",
    "validate_trades_frame",
    "safe_float_array",
]

