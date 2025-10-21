import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from market_nn_plus_ultra.service import ServiceSettings, create_app


def _build_sqlite_fixture(path: Path, *, rows: int = 64) -> None:
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(rows)]
    price_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * rows,
            "open": np.linspace(100, 120, rows),
            "high": np.linspace(101, 121, rows),
            "low": np.linspace(99, 119, rows),
            "close": np.linspace(100.5, 120.5, rows),
            "volume": np.linspace(1_000, 2_000, rows),
        }
    )
    assets_df = pd.DataFrame(
        {
            "asset_id": [1],
            "symbol": ["TEST"],
            "sector": ["tech"],
            "currency": ["USD"],
            "exchange": ["SIM"],
            "metadata": ["{}"],
        }
    )

    with sqlite3.connect(path) as conn:
        price_df.to_sql("series", conn, index=False, if_exists="replace")
        assets_df.to_sql("assets", conn, index=False, if_exists="replace")


def _write_config(path: Path, db_path: Path) -> None:
    config = {
        "seed": 7,
        "data": {
            "sqlite_path": str(db_path),
            "symbol_universe": ["TEST"],
            "feature_set": [],
            "window_size": 16,
            "horizon": 4,
            "stride": 4,
            "normalise": True,
            "val_fraction": 0.0,
        },
        "model": {
            "feature_dim": 5,
            "model_dim": 64,
            "depth": 2,
            "heads": 4,
            "dropout": 0.1,
            "conv_kernel_size": 3,
            "conv_dilations": [1, 2],
            "horizon": 4,
            "output_dim": 1,
            "architecture": "temporal_transformer",
        },
        "optimizer": {"lr": 1e-3},
        "trainer": {
            "batch_size": 8,
            "num_workers": 0,
            "accelerator": "cpu",
            "precision": "32-true",
            "checkpoint_dir": str(path.parent / "checkpoints"),
            "max_epochs": 5,
        },
        "guardrails": {
            "enabled": True,
            "capital_base": 400.0,
            "max_gross_exposure": 0.6,
            "max_symbol_exposure": 0.6,
            "sector_caps": {"tech": 0.5},
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_service_endpoints(tmp_path: Path) -> None:
    db_path = tmp_path / "market_fixture.db"
    _build_sqlite_fixture(db_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, db_path)

    settings = ServiceSettings(config_path=config_path, device="cpu", max_prediction_rows=32)
    app = create_app(settings)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["config_path"] == str(config_path)

    config_resp = client.get("/config")
    assert config_resp.status_code == 200
    config_payload = config_resp.json()
    assert config_payload["config"]["data"]["sqlite_path"] == str(db_path)

    curriculum_resp = client.get("/curriculum", params={"epochs": 3})
    assert curriculum_resp.status_code == 200
    curriculum_payload = curriculum_resp.json()
    assert curriculum_payload["epochs"] == 3
    assert len(curriculum_payload["stages"]) == 3

    predict_resp = client.post("/predict", json={"limit": 5})
    assert predict_resp.status_code == 200
    predict_payload = predict_resp.json()
    assert 0 < predict_payload["rows"] <= 5
    assert len(predict_payload["predictions"]) == predict_payload["rows"]
    assert predict_payload["telemetry"]["horizon"] == 4
    assert len(predict_payload["telemetry"]["feature_columns"]) == 5
    assert predict_payload["metrics"] is not None
    guardrail_info = predict_payload["telemetry"].get("guardrails")
    assert guardrail_info is not None
    assert guardrail_info["enabled"] is True
    assert guardrail_info["max_gross_exposure"] == pytest.approx(0.6)

    reload_resp = client.post("/reload", json={})
    assert reload_resp.status_code == 200
    reload_payload = reload_resp.json()
    assert reload_payload["checkpoint_path"] is None

    trades = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "symbol": "TEST",
            "sector": "tech",
            "notional": 320.0,
            "price": 10.0,
            "position": 32.0,
            "pnl": -12.0,
        },
        {
            "timestamp": "2024-01-01T01:00:00Z",
            "symbol": "TEST",
            "sector": "tech",
            "notional": 280.0,
            "price": 10.0,
            "position": 28.0,
            "pnl": -8.0,
        },
    ]
    guardrail_resp = client.post("/guardrails", json={"trades": trades})
    assert guardrail_resp.status_code == 200
    guardrail_payload = guardrail_resp.json()
    assert guardrail_payload["scaled"] is True
    assert guardrail_payload["violations"] == []
    assert guardrail_payload["metrics"]["gross_exposure_peak"] <= 0.6 + 1e-6

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_body = metrics_resp.text
    assert "plus_ultra_service_requests_total" in metrics_body
    assert "plus_ultra_guardrails_enabled" in metrics_body
