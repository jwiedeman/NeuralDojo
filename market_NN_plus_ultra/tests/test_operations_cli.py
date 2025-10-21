"""Regression coverage for the operations summary CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import operations_summary


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_operations_summary_cli_without_trades(tmp_path: Path, capsys) -> None:
    predictions = pd.DataFrame(
        {
            "window_end": pd.date_range("2025-01-01", periods=5, freq="D"),
            "realised_return": [0.02, -0.01, 0.015, 0.005, 0.01],
        }
    )
    predictions_path = _write_frame(predictions, tmp_path / "predictions.csv")

    exit_code = operations_summary.main([
        "--predictions",
        str(predictions_path),
        "--indent",
        "0",
    ])

    assert exit_code == 0
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["risk"]["sharpe"] != 0.0
    assert payload.get("guardrails") is None
    assert payload.get("triggered") == []


def test_operations_summary_cli_with_trades_and_thresholds(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "window_end": pd.date_range("2025-01-01", periods=5, freq="D"),
            "realised_return": [0.02, -0.03, 0.015, -0.025, 0.01],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
            "symbol": ["AAA", "AAA", "BBB", "BBB", "BBB"],
            "position": [100, -80, 120, -90, 140],
            "price": [10.0, 10.5, 9.5, 9.8, 10.2],
            "pnl": [200.0, -260.0, 150.0, -220.0, 175.0],
        }
    )
    predictions_path = _write_frame(predictions, tmp_path / "predictions.csv")
    trades_path = _write_frame(trades, tmp_path / "trades.csv")
    output_path = tmp_path / "summary.json"

    exit_code = operations_summary.main(
        [
            "--predictions",
            str(predictions_path),
            "--trades",
            str(trades_path),
            "--output",
            str(output_path),
            "--capital-base",
            "1000",
            "--min-sharpe",
            "1.5",
            "--max-drawdown",
            "0.01",
            "--max-gross-exposure",
            "0.4",
            "--max-turnover",
            "0.05",
            "--min-tail-return",
            "-150",
            "--max-tail-frequency",
            "0.1",
            "--max-symbol-exposure",
            "0.5",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["guardrails"]["gross_exposure_peak"] > 0.0
    assert payload["triggered"], "expected threshold alerts to be emitted"
    assert any("Sharpe" in message for message in payload["triggered"])

