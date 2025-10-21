from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import walkforward_eval


def _build_predictions(tmp_path: Path) -> Path:
    timestamps = pd.date_range("2024-01-01", periods=16, freq="D")
    returns = pd.Series(range(len(timestamps)), dtype=float) / 1_000.0
    frame = pd.DataFrame({"window_end": timestamps, "realised_return": returns})
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_walkforward_eval_script_produces_outputs(tmp_path: Path) -> None:
    predictions = _build_predictions(tmp_path)
    metrics_output = tmp_path / "metrics.parquet"
    summary_output = tmp_path / "summary.json"

    exit_code = walkforward_eval.main(
        [
            "--predictions",
            str(predictions),
            "--train-window",
            "8",
            "--test-window",
            "4",
            "--metrics-output",
            str(metrics_output),
            "--summary-output",
            str(summary_output),
            "--quiet",
        ]
    )

    assert exit_code == 0
    metrics = pd.read_parquet(metrics_output)
    assert not metrics.empty
    assert {"metric_sharpe", "metric_max_drawdown"}.issubset(metrics.columns)

    payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert payload["splits"] == len(metrics)
    assert "sharpe" in payload["metrics"]
    assert payload["metrics"]["sharpe"]["mean"] is not None
    assert "best_sharpe_split" in payload
