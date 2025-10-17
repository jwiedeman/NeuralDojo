import sqlite3
from datetime import datetime
from pathlib import Path

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from market_nn_plus_ultra.cli.dataset_build import main as dataset_build_main
from market_nn_plus_ultra.data.fixtures import FixtureConfig, build_fixture, write_fixture


@pytest.fixture()
def synthetic_db(tmp_path: Path) -> Path:
    config = FixtureConfig(
        symbols=["ALPHA", "BETA", "GAMMA"],
        rows=256,
        freq="30min",
        seed=7,
        start=datetime(2021, 1, 1),
        alt_features=2,
    )
    frames = build_fixture(config)
    db_path = write_fixture(frames, tmp_path / "fixture.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS regimes")
        conn.commit()
    return db_path


def _load_regimes(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM regimes", conn, parse_dates=["timestamp"])


def _load_cross_asset_view(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM cross_asset_views", conn, parse_dates=["timestamp"])


def test_dataset_build_cli_populates_regimes_table(synthetic_db: Path) -> None:
    exit_code = dataset_build_main(
        [
            str(synthetic_db),
            "--regime-labels",
            "--regime-bands",
            "volatility:0.25,0.75",
            "--regime-bands",
            "liquidity:0.2,0.8",
            "--strict-validation",
        ]
    )
    assert exit_code == 0

    regimes = _load_regimes(synthetic_db)
    assert set(regimes["name"].unique()) == {
        "volatility_regime",
        "liquidity_regime",
        "rotation_role",
    }
    assert not regimes.duplicated(["timestamp", "symbol", "name"]).any()

    per_symbol_counts = regimes.groupby(["symbol", "name"]).size().unstack("name")
    assert (per_symbol_counts > 0).all().all()


def test_dataset_build_cli_supports_output_copy(synthetic_db: Path, tmp_path: Path) -> None:
    target = tmp_path / "copy.db"
    exit_code = dataset_build_main(
        [
            str(synthetic_db),
            "--output",
            str(target),
            "--regime-labels",
            "--regime-bands",
            "rotation:0.2,0.8",
        ]
    )
    assert exit_code == 0
    assert target.exists()

    original_regimes_exists = False
    with sqlite3.connect(synthetic_db) as conn:
        original_regimes_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='regimes'"
            ).fetchone()
        )
    assert not original_regimes_exists

    regimes = _load_regimes(target)
    assert not regimes.empty


def test_dataset_build_cli_rejects_invalid_band_string(synthetic_db: Path) -> None:
    exit_code = dataset_build_main(
        [
            str(synthetic_db),
            "--regime-bands",
            "volatility:0.9",  # missing upper bound
        ]
    )
    assert exit_code == 2


def test_dataset_build_cli_generates_cross_asset_view(synthetic_db: Path) -> None:
    exit_code = dataset_build_main(
        [
            str(synthetic_db),
            "--cross-asset-view",
            "--cross-asset-columns",
            "close",
            "volume",
        ]
    )
    assert exit_code == 0

    view = _load_cross_asset_view(synthetic_db)
    assert not view.empty
    assert {"timestamp", "feature", "value"}.issubset(view.columns)
    assert not view.duplicated(["timestamp", "feature"]).any()

    features = set(view["feature"].unique())
    assert any(feature.startswith("close__") for feature in features)
    assert any(feature.startswith("volume__") for feature in features)

    metadata = view["metadata"].dropna().iloc[0]
    payload = json.loads(metadata)
    assert "symbol" in payload and payload["symbol"]
