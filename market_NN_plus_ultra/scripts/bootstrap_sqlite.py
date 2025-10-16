"""Utility script to populate the Plus Ultra SQLite database with ETF data.

The pretraining and training loops expect a SQLite database at
``data/market.db`` (or whatever path is configured in the experiment YAML).
This helper downloads historical OHLCV candles for a basket of ETFs using
`yfinance`, then writes them into the schema documented in
``docs/sqlite_schema.md``.

Example usage::

    python scripts/bootstrap_sqlite.py --db-path data/market.db \
        --tickers SPY QQQ VTI IWM EFA EEM XLK XLF XLY XLP \
        --start 2000-01-01 --end 2025-01-01

The script is idempotent and can be re-run to refresh data. By default it
creates the database if it does not exist and upserts candles for the
requested tickers. Pass ``--overwrite`` to drop existing tables before
ingestion.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "VTI",
    "IWM",
    "EFA",
    "EEM",
    "XLK",
    "XLF",
    "XLY",
    "XLP",
    "XLV",
    "XLI",
    "XLE",
    "XLB",
    "VNQ",
    "TLT",
    "GLD",
    "SLV",
    "XLU",
    "SMH",
)


@dataclass(slots=True)
class BootstrapConfig:
    """Configuration captured from the CLI."""

    db_path: Path
    tickers: tuple[str, ...]
    start: date
    end: date
    overwrite: bool


def parse_args() -> BootstrapConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/market.db"),
        help="Location to create or update the SQLite database.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=list(DEFAULT_TICKERS),
        help="List of symbols to download (space separated).",
    )
    parser.add_argument(
        "--start",
        type=pd.Timestamp,
        default=pd.Timestamp("2000-01-01"),
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=pd.Timestamp,
        default=pd.Timestamp.today().normalize(),
        help="Exclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop existing tables before loading new data.",
    )
    args = parser.parse_args()
    tickers = tuple(str(t).upper() for t in args.tickers if str(t).strip())
    if not tickers:
        raise SystemExit("No tickers supplied; provide at least one symbol.")

    return BootstrapConfig(
        db_path=args.db_path,
        tickers=tickers,
        start=args.start.date(),
        end=args.end.date(),
        overwrite=args.overwrite,
    )


def ensure_schema(conn: sqlite3.Connection, overwrite: bool) -> None:
    if overwrite:
        conn.execute("DROP TABLE IF EXISTS series")
        conn.execute("DROP TABLE IF EXISTS assets")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            sector TEXT,
            currency TEXT,
            exchange TEXT,
            metadata TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS series (
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            vwap REAL,
            turnover REAL,
            PRIMARY KEY (timestamp, symbol),
            FOREIGN KEY (symbol) REFERENCES assets(symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_series_symbol_timestamp
            ON series(symbol, timestamp)
        """
    )


def upsert_assets(conn: sqlite3.Connection, tickers: Iterable[str]) -> None:
    metadata = json.dumps({"source": "yfinance"})
    conn.executemany(
        """
        INSERT INTO assets (symbol, currency, metadata)
        VALUES (?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            currency = excluded.currency,
            metadata = excluded.metadata
        """,
        ((symbol, "USD", metadata) for symbol in tickers),
    )


def fetch_ohlcv(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    data = yf.download(
        tickers=list(tickers),
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("No data returned from yfinance; check tickers or dates.")

    frames: list[pd.DataFrame] = []
    if isinstance(data.columns, pd.MultiIndex):
        for symbol in tickers:
            if symbol not in data.columns.get_level_values(0):
                continue
            sym_df = data.xs(symbol, axis=1)
            frames.append(_normalize_symbol_frame(sym_df, symbol))
    else:
        # Single ticker request returns a flat column index.
        frames.append(_normalize_symbol_frame(data, tickers[0]))

    if not frames:
        raise RuntimeError("No matching frames parsed from download response.")

    return pd.concat(frames, ignore_index=True)


def _normalize_symbol_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    renamed = (
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        .reset_index()
        .rename(columns={"Date": "timestamp"})
    )
    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=True)
    renamed["symbol"] = symbol
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    frame = renamed[columns].copy()
    frame["volume"] = frame["volume"].astype(float)
    return frame.dropna(subset=["open", "high", "low", "close"])


def upsert_series(conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
    records = [
        (
            ts.isoformat(),
            row.symbol,
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume) if pd.notna(row.volume) else None,
        )
        for ts, row in frame.set_index("timestamp").iterrows()
    ]
    conn.executemany(
        """
        INSERT INTO series (timestamp, symbol, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(timestamp, symbol) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume
        """,
        records,
    )


def main() -> None:
    config = parse_args()
    config.db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(config.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_schema(conn, config.overwrite)
        upsert_assets(conn, config.tickers)
        frame = fetch_ohlcv(config.tickers, config.start, config.end)
        upsert_series(conn, frame)
        conn.commit()

    print(
        "Loaded",
        len(config.tickers),
        "tickers and",
        len(frame),
        "candles into",
        config.db_path,
    )


if __name__ == "__main__":
    main()

