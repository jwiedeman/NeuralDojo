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
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from .validation import validate_indicator_frame, validate_price_frame


@dataclass(slots=True)
class SQLiteMarketSource:
    """Connection metadata for a market SQLite database."""

    path: str
    read_only: bool = True
    pragma_statements: Optional[Iterable[str]] = None

    def _resolve_path_candidates(self, raw_path: Path) -> list[Path]:
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append((Path.cwd() / raw_path).resolve())
            project_root = (Path(__file__).resolve().parents[2] / raw_path).resolve()
            if project_root not in candidates:
                candidates.append(project_root)
        return candidates

    def _ensure_filesystem_path(self, raw_path: Path) -> Path:
        candidates = self._resolve_path_candidates(raw_path.expanduser())
        for candidate in candidates:
            if candidate.exists():
                return candidate

        if not self.read_only and candidates:
            target = candidates[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            return target

        tried = ", ".join(str(c) for c in candidates) or str(raw_path)
        raise FileNotFoundError(
            "Could not locate the SQLite database. Checked: "
            f"{tried}. Provide a database at that location or update your "
            "configuration (see README.md step 'Provide market data')."
        )

    def _prepare_connection(self) -> tuple[str, Optional[Path]]:
        """Return a connection URI and resolved filesystem path (if any)."""

        if self.path == ":memory:":
            return self.path, None

        if self.path.startswith("sqlite:") or self.path.startswith("sqlite+"):
            return self.path, None

        if self.path.startswith("file:"):
            parsed = urlparse(self.path)
            path_str = parsed.path or parsed.netloc
            if not path_str:
                return self.path, None
            fs_path = self._ensure_filesystem_path(Path(unquote(path_str)))
            return self.path, fs_path

        fs_path = self._ensure_filesystem_path(Path(self.path))
        mode = "ro" if self.read_only else "rwc"
        uri = f"file:{fs_path.as_posix()}?mode={mode}"
        return uri, fs_path

    def connect(self) -> sqlite3.Connection:
        """Return a SQLite connection with pragmas applied."""

        uri, _ = self._prepare_connection()
        conn = sqlite3.connect(uri, uri=True, detect_types=sqlite3.PARSE_DECLTYPES)
        default_pragmas = ("journal_mode=WAL", "synchronous=NORMAL")
        pragmas = self.pragma_statements or default_pragmas
        for pragma in pragmas:
            if self.read_only and pragma.lower().startswith("journal_mode"):
                continue
            conn.execute(f"PRAGMA {pragma}")
        return conn

    def engine(self):
        """Return a SQLAlchemy engine for vectorised queries."""

        return create_engine("sqlite+pysqlite://", creator=self.connect)


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
    validate: bool = True

    def load(self) -> pd.DataFrame:
        """Return a multi-indexed frame with OHLCV and indicator columns."""

        with contextlib.closing(self.source.connect()) as conn:
            price_df = pd.read_sql_query("SELECT * FROM series", conn, parse_dates=["timestamp"])
            if self.validate:
                price_df = validate_price_frame(price_df)
            if self.symbol_universe:
                price_df = price_df[price_df["symbol"].isin(set(self.symbol_universe))]

            if self.tz_convert:
                price_df["timestamp"] = price_df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(self.tz_convert)

            price_df = price_df.set_index(["timestamp", "symbol"]).sort_index()

            indicator_frames: list[pd.DataFrame] = []
            for name, table in (self.indicators or {}).items():
                query = f"SELECT * FROM {table}" if " " not in table.lower() else table
                ind_df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
                if self.validate:
                    ind_df = validate_indicator_frame(ind_df)
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

