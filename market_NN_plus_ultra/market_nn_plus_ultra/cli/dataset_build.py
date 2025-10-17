"""Dataset assembly CLI for Market NN Plus Ultra.

This command wires Pandera-backed validation, optional market-regime
labelling, and SQLite plumbing into a single entrypoint so experimenters can
refresh datasets and smoke-test integrity from the terminal.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from ..data.cross_asset import build_cross_asset_view
from ..data.labelling import MarketRegimeLabellingConfig, generate_regime_labels
from ..data.validation import (
    DataValidationError,
    validate_assets_frame,
    validate_cross_asset_view_frame,
    validate_indicator_frame,
    validate_price_frame,
    validate_regime_frame,
    validate_sqlite_frames,
)
from ..utils.logging import StructuredLogger, get_structured_logger


@dataclass(frozen=True)
class RegimeBandOverride:
    """User-supplied quantile override for a regime labeller."""

    key: str
    lower: float
    upper: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-nn-plus-ultra-dataset-build",
        description="Validate a SQLite dataset and optionally regenerate regime labels.",
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the source SQLite database.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination SQLite file. Defaults to updating the input in-place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the destination database if it already exists.",
    )
    parser.add_argument(
        "--symbol-universe",
        nargs="+",
        help="Restrict operations to the provided ticker symbols before validation and labelling.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Run the full Pandera + relationship validation suite after mutations.",
    )
    parser.add_argument(
        "--regime-labels",
        action="store_true",
        help="Regenerate regime labels using the current price history.",
    )
    parser.add_argument(
        "--regime-bands",
        metavar="K:LOW,HIGH",
        action="append",
        type=_parse_regime_band,
        default=[],
        help=(
            "Override quantile bands for a regime labeller (e.g. "
            "'volatility:0.3,0.8'). Providing bands implies --regime-labels."
        ),
    )
    parser.add_argument(
        "--cross-asset-view",
        action="store_true",
        help="Generate the cross_asset_views table with aligned multi-symbol features.",
    )
    parser.add_argument(
        "--cross-asset-columns",
        nargs="+",
        default=["close", "volume"],
        help="Series columns to align when generating the cross-asset view (default: close volume).",
    )
    parser.add_argument(
        "--cross-asset-fill-limit",
        type=int,
        default=None,
        help="Maximum consecutive missing rows to forward/back-fill when aligning cross-asset tensors.",
    )
    parser.add_argument(
        "--cross-asset-no-returns",
        action="store_true",
        help="Disable automatic log-return features when building the cross-asset view.",
    )
    return parser


def _parse_regime_band(raw: str) -> RegimeBandOverride:
    try:
        key, bounds = raw.split(":", 1)
        lower_str, upper_str = bounds.split(",", 1)
    except ValueError as exc:  # pragma: no cover - argparse enforces usage
        raise argparse.ArgumentTypeError(
            "regime bands must be specified as 'name:lower,upper'"
        ) from exc

    key = key.strip().lower()
    if key not in {"volatility", "liquidity", "rotation"}:
        raise argparse.ArgumentTypeError(
            "regime band key must be one of 'volatility', 'liquidity', or 'rotation'"
        )

    try:
        lower = float(lower_str)
        upper = float(upper_str)
    except ValueError as exc:  # pragma: no cover - argparse enforces usage
        raise argparse.ArgumentTypeError("regime band bounds must be numeric") from exc

    if not 0.0 <= lower <= 1.0 or not 0.0 <= upper <= 1.0:
        raise argparse.ArgumentTypeError("regime band bounds must be within [0, 1]")
    if lower >= upper:
        raise argparse.ArgumentTypeError("regime band lower bound must be < upper bound")

    return RegimeBandOverride(key=key, lower=lower, upper=upper)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
    return cursor.fetchone() is not None


def _resolve_destination(source: Path, output: Path | None, *, overwrite: bool) -> Path:
    if output is None:
        return source

    destination = output
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"destination '{destination}' exists; pass --overwrite to replace it"
            )
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _subset_assets(assets: pd.DataFrame | None, symbols: set[str] | None) -> pd.DataFrame | None:
    if assets is None or not symbols:
        return assets
    return assets[assets["symbol"].isin(symbols)].reset_index(drop=True)


def _prepare_price_frame(
    price_df: pd.DataFrame,
    *,
    symbols: set[str] | None,
    strict: bool,
    assets: pd.DataFrame | None,
    logger: StructuredLogger,
) -> pd.DataFrame:
    if symbols:
        price_df = price_df[price_df["symbol"].isin(symbols)]
    if price_df.empty:
        raise ValueError("no price rows remain after applying filters")

    if strict:
        return validate_price_frame(price_df, assets=assets, logger=logger)

    prepared = price_df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=False)
    prepared["symbol"] = prepared["symbol"].astype(str)
    return prepared.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _prepare_assets_frame(
    assets: pd.DataFrame | None,
    *,
    strict: bool,
    logger: StructuredLogger,
) -> pd.DataFrame | None:
    if assets is None:
        return None
    if strict:
        return validate_assets_frame(assets, logger=logger)
    prepared = assets.copy()
    prepared["symbol"] = prepared["symbol"].astype(str)
    return prepared.reset_index(drop=True)


def _apply_regime_overrides(
    overrides: Iterable[RegimeBandOverride],
    base: MarketRegimeLabellingConfig,
) -> MarketRegimeLabellingConfig:
    config = base
    for override in overrides:
        if override.key == "volatility":
            config.volatility.lower_quantile = override.lower
            config.volatility.upper_quantile = override.upper
        elif override.key == "liquidity":
            config.liquidity.lower_quantile = override.lower
            config.liquidity.upper_quantile = override.upper
        else:
            config.rotation.lower_quantile = override.lower
            config.rotation.upper_quantile = override.upper
    return config


def _validate_related_tables(
    conn: sqlite3.Connection,
    *,
    assets: pd.DataFrame | None,
    strict: bool,
    logger: StructuredLogger,
) -> Mapping[str, pd.DataFrame]:
    frames: MutableMapping[str, pd.DataFrame] = {}
    if assets is not None:
        frames["assets"] = assets

    if _table_exists(conn, "series"):
        price_df = pd.read_sql_query("SELECT * FROM series", conn, parse_dates=["timestamp"])
        frames["series"] = (
            validate_price_frame(price_df, assets=assets, logger=logger) if strict else price_df
        )

    if _table_exists(conn, "indicators"):
        indicator_df = pd.read_sql_query("SELECT * FROM indicators", conn, parse_dates=["timestamp"])
        frames["indicators"] = (
            validate_indicator_frame(indicator_df, assets=assets, logger=logger)
            if strict
            else indicator_df
        )

    if _table_exists(conn, "regimes"):
        regime_df = pd.read_sql_query("SELECT * FROM regimes", conn, parse_dates=["timestamp"])
        frames["regimes"] = (
            validate_regime_frame(regime_df, assets=assets, logger=logger)
            if strict
            else regime_df
        )

    if _table_exists(conn, "cross_asset_views"):
        cross_asset_df = pd.read_sql_query(
            "SELECT * FROM cross_asset_views", conn, parse_dates=["timestamp"]
        )
        frames["cross_asset_views"] = (
            validate_cross_asset_view_frame(cross_asset_df, logger=logger)
            if strict
            else cross_asset_df
        )

    if strict:
        validate_sqlite_frames(frames, logger=logger)
    return frames


def run(args: argparse.Namespace, logger: StructuredLogger) -> None:
    source = args.db_path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"database '{source}' does not exist")

    destination = _resolve_destination(source, args.output.resolve() if args.output else None, overwrite=args.overwrite)

    logger.info(
        "dataset_build_start",
        source=str(source),
        destination=str(destination),
        regime_labels=bool(args.regime_labels or args.regime_bands),
        strict_validation=bool(args.strict_validation),
        symbol_universe=args.symbol_universe or [],
        cross_asset_view=bool(args.cross_asset_view),
        cross_asset_columns=args.cross_asset_columns if args.cross_asset_view else [],
        cross_asset_fill_limit=args.cross_asset_fill_limit,
        cross_asset_returns=not args.cross_asset_no_returns,
    )

    with sqlite3.connect(destination) as conn:
        if not _table_exists(conn, "series"):
            raise ValueError("the SQLite database must contain a 'series' table")

        assets_df: pd.DataFrame | None = None
        if _table_exists(conn, "assets"):
            assets_df = pd.read_sql_query("SELECT * FROM assets", conn)
        assets_df = _prepare_assets_frame(assets_df, strict=args.strict_validation, logger=logger)

        symbols = set(args.symbol_universe) if args.symbol_universe else None
        price_df = pd.read_sql_query("SELECT * FROM series", conn, parse_dates=["timestamp"])
        price_df = _prepare_price_frame(
            price_df,
            symbols=symbols,
            strict=args.strict_validation,
            assets=assets_df,
            logger=logger,
        )

        if symbols and assets_df is not None:
            assets_df = _subset_assets(assets_df, symbols)

        regime_requested = bool(args.regime_labels or args.regime_bands)
        if regime_requested:
            config = _apply_regime_overrides(args.regime_bands, MarketRegimeLabellingConfig())
            regimes = generate_regime_labels(price_df, config=config, assets=assets_df, logger=logger)
            regimes.to_sql("regimes", conn, index=False, if_exists="replace")
            logger.info(
                "regime_labels_written",
                rows=int(len(regimes)),
                symbols=sorted(regimes["symbol"].unique()),
                labels=sorted(regimes["name"].unique()),
            )

        if args.cross_asset_view:
            result = build_cross_asset_view(
                price_df,
                value_columns=args.cross_asset_columns,
                include_returns=not args.cross_asset_no_returns,
                fill_limit=args.cross_asset_fill_limit,
                universe_name=(
                    ",".join(sorted(assets_df["symbol"].astype(str))) if assets_df is not None else None
                ),
                logger=logger,
            )
            cross_asset_df = validate_cross_asset_view_frame(result.frame, logger=logger)
            cross_asset_df.to_sql("cross_asset_views", conn, index=False, if_exists="replace")
            logger.info(
                "cross_asset_view_written",
                rows=int(len(cross_asset_df)),
                features=result.stats.feature_columns,
                fill_rate=float(result.stats.fill_rate),
                dropped_rows=int(result.stats.dropped_rows),
                dropped_features=list(result.stats.dropped_features),
                universe=list(result.stats.universe),
            )

        _validate_related_tables(conn, assets=assets_df, strict=args.strict_validation, logger=logger)

    logger.info("dataset_build_complete", destination=str(destination))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - argparse already reported the error
        code = exc.code if isinstance(exc.code, int) else 1
        return code

    logger = get_structured_logger("market_nn_plus_ultra.cli.dataset_build")
    try:
        run(args, logger)
    except FileNotFoundError as exc:
        logger.error("dataset_build_failed", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, DataValidationError) as exc:
        logger.error("dataset_build_failed", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = ["main", "run"]
