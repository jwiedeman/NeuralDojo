"""SQLite ingestion utilities for Market NN Plus Ultra.

This module loads OHLCV series, optional indicator tables, and auxiliary
features from a SQLite database. It exposes a high-level dataset helper that
returns a tidy multi-indexed :class:`pandas.DataFrame` ready for feature
engineering and model consumption.
"""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


@dataclass(slots=True)
class SQLiteMarketSource:
    """Connection metadata for a market SQLite database."""

    path: str
    read_only: bool = True
    pragma_statements: Optional[Iterable[str]] = None

    def connect(self) -> sqlite3.Connection:
        """Return a SQLite connection with pragmas applied."""

        uri = f"file:{self.path}?mode={'ro' if self.read_only else 'rwc'}"
        conn = sqlite3.connect(uri, uri=True, detect_types=sqlite3.PARSE_DECLTYPES)
        for pragma in self.pragma_statements or ("journal_mode=WAL", "synchronous=NORMAL"):
            conn.execute(f"PRAGMA {pragma}")
        return conn

    def engine(self):
        """Return a SQLAlchemy engine for vectorised queries."""

        mode = "ro" if self.read_only else "rw"
        return create_engine(f"sqlite+pysqlite:///{self.path}?mode={mode}")


@dataclass(slots=True)
class SQLiteMarketDataset:
    """Load structured market data from a SQLite database.

    Parameters
    ----------
    source:
        Database connection metadata.
    symbol_universe:
        Optional list of ticker symbols to filter.
    indicators:
        Mapping of indicator name to SQL query or table name.
    resample_rule:
        Optional pandas offset alias for resampling (e.g., ``'1H'``).
    tz_convert:
        Optional timezone to convert the ``timestamp`` column into.
    """

    source: SQLiteMarketSource
    symbol_universe: Optional[Iterable[str]] = None
    indicators: Optional[Mapping[str, str]] = None
    resample_rule: Optional[str] = None
    tz_convert: Optional[str] = None

    def load(self) -> pd.DataFrame:
        """Return a multi-indexed frame with OHLCV and indicator columns."""

        with contextlib.closing(self.source.connect()) as conn:
            price_df = pd.read_sql_query("SELECT * FROM series", conn, parse_dates=["timestamp"])
            if self.symbol_universe:
                price_df = price_df[price_df["symbol"].isin(set(self.symbol_universe))]

            if self.tz_convert:
                price_df["timestamp"] = price_df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(self.tz_convert)

            price_df = price_df.set_index(["timestamp", "symbol"]).sort_index()

            indicator_frames: list[pd.DataFrame] = []
            for name, table in (self.indicators or {}).items():
                query = f"SELECT * FROM {table}" if " " not in table.lower() else table
                ind_df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
                ind_df = ind_df.set_index(["timestamp", "symbol"]).sort_index()
                indicator_frames.append(ind_df.add_prefix(f"{name}__"))

        if indicator_frames:
            merged = pd.concat([price_df] + indicator_frames, axis=1)
        else:
            merged = price_df

        if self.resample_rule:
            merged = (
                merged.groupby(level="symbol")
                .apply(lambda df: df.droplevel("symbol").resample(self.resample_rule).agg("last"))
                .drop(columns=["symbol"], errors="ignore")
            )
            merged.index.names = ["symbol", "timestamp"]
            merged = merged.swaplevel().sort_index()

        merged = merged.dropna(how="all")
        return merged

    def as_panel(self) -> pd.DataFrame:
        """Return the dataset as a panel with contiguous timestamps per symbol."""

        df = self.load()
        symbols = df.index.get_level_values("symbol").unique()
        frames = []
        for sym in symbols:
            sym_df = df.xs(sym, level="symbol").copy()
            sym_df = sym_df.ffill().bfill()
            sym_df["symbol"] = sym
            frames.append(sym_df)
        panel = pd.concat(frames)
        panel.index.name = "timestamp"
        return panel.reset_index().set_index(["timestamp", "symbol"]).sort_index()

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return timestamps, data matrix, and feature names."""

        panel = self.as_panel()
        feature_cols = [c for c in panel.columns if c not in ("symbol",)]
        timestamps = panel.index.get_level_values("timestamp").unique().to_numpy()
        data = panel[feature_cols].to_numpy(dtype=np.float32)
        return timestamps, data, feature_cols

