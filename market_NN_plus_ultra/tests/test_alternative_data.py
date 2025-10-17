from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from market_nn_plus_ultra.data import AlternativeDataSpec, SQLiteMarketDataset, SQLiteMarketSource


def _seed_database(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE series (
                timestamp TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            );
            CREATE TABLE macro_calendar (
                timestamp TEXT,
                impact REAL
            );
            CREATE TABLE funding_rates (
                timestamp TEXT,
                symbol TEXT,
                rate REAL
            );
            """
        )
        series_rows = [
            ("2024-01-01T00:00:00", "BTC", 1, 1, 1, 1, 100),
            ("2024-01-02T00:00:00", "BTC", 2, 2, 2, 2, 110),
            ("2024-01-03T00:00:00", "BTC", 3, 3, 3, 3, 120),
            ("2024-01-01T00:00:00", "ETH", 4, 4, 4, 4, 200),
            ("2024-01-02T00:00:00", "ETH", 5, 5, 5, 5, 210),
            ("2024-01-03T00:00:00", "ETH", 6, 6, 6, 6, 220),
        ]
        conn.executemany(
            "INSERT INTO series (timestamp, symbol, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            series_rows,
        )
        conn.executemany(
            "INSERT INTO macro_calendar (timestamp, impact) VALUES (?, ?)",
            [
                ("2024-01-02T00:00:00", 3.0),
            ],
        )
        conn.executemany(
            "INSERT INTO funding_rates (timestamp, symbol, rate) VALUES (?, ?, ?)",
            [
                ("2024-01-01T00:00:00", "BTC", 0.01),
                ("2024-01-02T00:00:00", "ETH", 0.02),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_alternative_data_connectors(tmp_path) -> None:
    db_path = tmp_path / "alt_data.db"
    _seed_database(str(db_path))

    source = SQLiteMarketSource(path=str(db_path))
    dataset = SQLiteMarketDataset(
        source=source,
        alternative_data=[
            AlternativeDataSpec(
                name="macro",
                table="macro_calendar",
                join_columns=("timestamp",),
                columns=("impact",),
                prefix="macro",
                fill_forward=True,
            ),
            AlternativeDataSpec(
                name="funding",
                table="funding_rates",
                join_columns=("timestamp", "symbol"),
                columns=("rate",),
                prefix="funding",
                fill_forward=True,
            ),
        ],
        validate=False,
    )

    panel = dataset.load()

    assert "macro__impact" in panel.columns
    assert "funding__rate" in panel.columns

    btc_third = panel.loc[(pd.Timestamp("2024-01-03T00:00:00"), "BTC"), "funding__rate"]
    eth_third = panel.loc[(pd.Timestamp("2024-01-03T00:00:00"), "ETH"), "funding__rate"]
    macro_value = panel.loc[(pd.Timestamp("2024-01-02T00:00:00"), "BTC"), "macro__impact"]

    assert pytest.approx(btc_third, rel=1e-6) == 0.01
    assert pytest.approx(eth_third, rel=1e-6) == 0.02
    assert pytest.approx(macro_value, rel=1e-6) == 3.0

