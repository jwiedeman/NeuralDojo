import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_nn_plus_ultra.monitoring import (
    DriftAlertThresholds,
    LiveMonitor,
    MonitoringSnapshot,
    compute_drift_metrics,
)
from market_nn_plus_ultra.monitoring.prometheus import make_prometheus_exporter


def test_compute_drift_metrics_detects_shift():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0.0, scale=1.0, size=1024)
    observed = rng.normal(loc=0.5, scale=1.2, size=1024)

    metrics = compute_drift_metrics(reference, observed, bins=30)

    assert metrics.population_stability_index > 0.0
    assert metrics.js_divergence > 0.0
    assert 0.0 <= metrics.ks_statistic <= 1.0


def test_live_monitor_emits_alerts(tmp_path: Path):
    reference = np.concatenate((np.full(128, 0.01), np.full(128, -0.005)))
    live = np.full(128, -0.02)

    monitor = LiveMonitor(
        reference,
        window_size=128,
        risk_thresholds=None,
        drift_thresholds=DriftAlertThresholds(psi_alert=0.01, js_alert=0.001, ks_alert=0.01),
    )
    snapshot = monitor.update(live)

    assert isinstance(snapshot, MonitoringSnapshot)
    assert snapshot.window_count == 128
    assert snapshot.risk["sharpe"] <= 0.0
    assert any("Population stability index" in alert for alert in snapshot.alerts)


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_live_monitor_cli_runs(tmp_path: Path, suffix: str):
    reference = pd.DataFrame({"realised_return": np.linspace(-0.01, 0.02, 32)})
    live = pd.DataFrame({"realised_return": np.linspace(-0.05, 0.01, 32)})

    reference_path = tmp_path / f"reference{suffix}"
    live_path = tmp_path / f"live{suffix}"

    if suffix == ".csv":
        reference.to_csv(reference_path, index=False)
        live.to_csv(live_path, index=False)
    else:
        reference.to_parquet(reference_path)
        live.to_parquet(live_path)

    output = tmp_path / "snapshot.json"

    from market_NN_plus_ultra.scripts.monitoring.live_monitor import main

    code = main(
        [
            str(reference_path),
            str(live_path),
            "--window",
            "32",
            "--drift-bins",
            "10",
            "--psi-alert",
            "0.0",
            "--js-alert",
            "0.0",
            "--ks-alert",
            "0.0",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text())
    assert payload["window_count"] == 32
    assert set(payload["risk"]).issuperset({"sharpe", "max_drawdown"})


def test_live_monitor_cli_with_operations_summary(tmp_path: Path):
    reference = pd.DataFrame({"realised_return": np.linspace(-0.01, 0.02, 16)})
    predictions = pd.DataFrame({"realised_return": np.linspace(-0.01, 0.02, 16)})

    reference_path = tmp_path / "reference.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    reference.to_parquet(reference_path)
    predictions.to_parquet(predictions_path)

    operations_payload = {
        "risk": {"roi": 0.0125, "sharpe": 1.2, "max_drawdown": -0.04},
        "guardrails": {"gross_exposure_peak": 1.1},
        "triggered": ["Gross exposure peak 1.100 exceeded limit 1.000"],
    }
    operations_path = tmp_path / "operations.json"
    operations_path.write_text(json.dumps(operations_payload), encoding="utf-8")

    output = tmp_path / "snapshot.json"

    from market_NN_plus_ultra.scripts.monitoring.live_monitor import main

    code = main(
        [
            str(reference_path),
            "--predictions",
            str(predictions_path),
            "--operations-summary",
            str(operations_path),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text())
    assert pytest.approx(payload["risk"]["roi"]) == operations_payload["risk"]["roi"]
    assert payload["alerts"] == operations_payload["triggered"]
    assert pytest.approx(payload["guardrails"]["gross_exposure_peak"]) == operations_payload["guardrails"][
        "gross_exposure_peak"
    ]


def test_prometheus_exporter_updates_metrics():
    prometheus_client = pytest.importorskip("prometheus_client")
    registry = prometheus_client.CollectorRegistry()
    exporter = make_prometheus_exporter(registry)

    snapshot = MonitoringSnapshot(
        risk={"sharpe": 0.42, "max_drawdown": -0.05},
        drift=compute_drift_metrics([0.0, 0.1], [0.2, 0.3]),
        alerts=["Sharpe ratio 0.420 below threshold 0.500"],
        window_count=64,
    )

    exporter.publish(snapshot)

    metrics = {}
    for metric in registry.collect():
        for sample in metric.samples:
            metrics[sample.name] = sample.value

    assert metrics["plus_ultra_monitoring_risk_metric"] >= 0.0
    assert metrics["plus_ultra_monitoring_window_size"] == 64.0
    assert metrics["plus_ultra_monitoring_alerts_total"] == 1.0
