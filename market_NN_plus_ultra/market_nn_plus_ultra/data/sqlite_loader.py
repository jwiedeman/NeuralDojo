"""Utilities for loading market data from a SQLite database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine, text


@dataclass(slots=True)
class SQLiteMarketDataset:
    """Loads asset time-series and indicators from a SQLite database.

    The database is expected to expose the following tables:

    * ``assets``: ``asset_id`` (int), ``symbol`` (str), ``sector`` (str, optional).
    * ``series``: ``timestamp`` (int / str), ``asset_id`` (int), ``open``, ``high``, ``low``, ``close``, ``volume``.
    * ``indicators``: ``timestamp`` (int / str), ``asset_id`` (int), ``name`` (str), ``value`` (float).
    * ``meta`` (optional): global metadata such as currency or data source.

    Additional tables such as ``trades`` and ``benchmarks`` can be introduced later for
    reinforcement learning fine-tuning.
    """

    database_path: Path
    indicators: Optional[Iterable[str]] = None
    asset_universe: Optional[Iterable[str]] = None

    def _engine(self):
        return create_engine(f"sqlite:///{self.database_path}")

    def load_prices(self) -> pd.DataFrame:
        """Load OHLCV price history for the requested asset universe."""

        query = "SELECT * FROM series"
        if self.asset_universe:
            placeholders = ",".join([":asset_" + str(i) for i, _ in enumerate(self.asset_universe)])
            query += f" WHERE asset_id IN (SELECT asset_id FROM assets WHERE symbol IN ({placeholders}))"
        with self._engine().connect() as conn:
            params = {f"asset_{i}": symbol for i, symbol in enumerate(self.asset_universe or [])}
            return pd.read_sql(text(query), conn, params=params, parse_dates=["timestamp"]).sort_values(
                ["asset_id", "timestamp"],
            )

    def load_indicators(self) -> pd.DataFrame:
        """Load indicator panel data filtered by names if provided."""

        query = "SELECT * FROM indicators"
        params: dict[str, object] = {}
        if self.indicators:
            placeholders = ",".join([":name_" + str(i) for i, _ in enumerate(self.indicators)])
            query += f" WHERE name IN ({placeholders})"
            params = {f"name_{i}": name for i, name in enumerate(self.indicators)}
        with self._engine().connect() as conn:
            return pd.read_sql(text(query), conn, params=params, parse_dates=["timestamp"]).sort_values(
                ["asset_id", "timestamp", "name"],
            )

    def load_joined_panel(self) -> pd.DataFrame:
        """Return a multi-indexed dataframe combining prices and indicators."""

        prices = self.load_prices()
        indicators = self.load_indicators()
        pivoted = indicators.pivot_table(
            index=["timestamp", "asset_id"],
            columns="name",
            values="value",
        )
        merged = prices.merge(pivoted, on=["timestamp", "asset_id"], how="left")
        merged = merged.sort_values(["asset_id", "timestamp"]).set_index(["asset_id", "timestamp"])
        return merged
