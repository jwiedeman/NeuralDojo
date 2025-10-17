from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from market_nn_plus_ultra.data.fixtures import FixtureConfig, build_fixture, write_fixture


def test_build_fixture_generates_expected_tables(tmp_path: Path) -> None:
    config = FixtureConfig(
        symbols=["ALPHA", "BETA"],
        rows=512,
        freq="30min",
        seed=42,
        start=datetime(2020, 1, 1),
        alt_features=2,
    )
    frames = build_fixture(config)

    assert set(frames) == {"series", "indicators", "regimes", "assets"}

    price_df = frames["series"]
    assert len(price_df) == config.rows * len(config.symbols)
    assert {"open", "high", "low", "close", "volume", "turnover"}.issubset(price_df.columns)
    for _, group in price_df.groupby("symbol"):
        assert group["timestamp"].is_monotonic_increasing

    indicator_df = frames["indicators"]
    assert {"ma_24", "ma_96", "alt_signal_1"}.issubset(set(indicator_df["name"].unique()))

    db_path = write_fixture(frames, tmp_path / "fixture.db")
    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"series", "indicators", "regimes", "assets"}.issubset(tables)
        row_count = conn.execute("SELECT COUNT(*) FROM series").fetchone()[0]
        assert row_count == config.rows * len(config.symbols)

        sample = pd.read_sql_query("SELECT * FROM indicators LIMIT 5", conn)
        assert not sample.empty
