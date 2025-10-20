"""Regression coverage for the architecture sweep CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_nn_plus_ultra.training import BenchmarkScenario, TrainingRunResult


def _load_architecture_sweep() -> object:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "architecture_sweep.py"
    spec = importlib.util.spec_from_file_location("architecture_sweep", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ConfigStub:
    def __init__(self, label: str) -> None:
        self.label = label


@pytest.fixture()
def expected_records() -> list[dict[str, object]]:
    fixture_path = Path(__file__).parent / "fixtures" / "architecture_sweep_expected.json"
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_architecture_sweep_generates_stable_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected_records: list[dict[str, object]]) -> None:
    """The CLI should persist deterministic parquet outputs when training is stubbed."""

    architecture_sweep = _load_architecture_sweep()

    base_config = SimpleNamespace(
        model=SimpleNamespace(
            architecture="omni_mixture",
            model_dim=768,
            depth=12,
            horizon=5,
            conv_dilations=(1, 2, 4, 8),
        ),
        data=SimpleNamespace(horizon=5),
    )

    scenarios = [
        BenchmarkScenario("omni_mixture", 768, 12, 5, (1, 2, 4, 8), label="omni-gpu"),
        BenchmarkScenario("hybrid_transformer", 512, 8, 5, (1, 3, 9, 27), label="hybrid-gpu"),
        BenchmarkScenario("state_space", 256, 6, 5, (1, 2, 2, 2), label="baseline-cpu"),
    ]

    perf_counter_values = iter([10.0, 10.1, 20.0, 20.25, 30.0, 30.22])

    def fake_perf_counter() -> float:
        return next(perf_counter_values)

    def fake_load_experiment(_: Path) -> SimpleNamespace:
        return base_config

    def fake_iter_scenarios(*_args, **_kwargs):
        return iter(scenarios)

    def fake_prepare_config(_config, scenario, **_kwargs):
        return _ConfigStub(scenario.label or scenario.architecture)

    def fake_run_training(config: _ConfigStub) -> TrainingRunResult:
        label = config.label
        metrics = {
            "val/loss": {
                "omni-gpu": 0.1,
                "hybrid-gpu": 0.2,
                "baseline-cpu": 0.3,
            }[label],
            "train/loss": {
                "omni-gpu": 0.08,
                "hybrid-gpu": 0.15,
                "baseline-cpu": 0.25,
            }[label],
        }
        return TrainingRunResult(
            best_model_path=f"artifacts/{label}.ckpt",
            logged_metrics=metrics,
            dataset_summary={"samples": 64, "batches": 4},
            profitability_summary={"roi": 0.1, "sharpe": 1.0, "max_drawdown": 0.2},
            profitability_reports={"json": f"artifacts/{label}-profit.json"},
        )

    monkeypatch.setattr(architecture_sweep, "load_experiment_from_file", fake_load_experiment)
    monkeypatch.setattr(architecture_sweep, "iter_scenarios", fake_iter_scenarios)
    monkeypatch.setattr(architecture_sweep, "prepare_config_for_scenario", fake_prepare_config)
    monkeypatch.setattr(architecture_sweep, "run_training", fake_run_training)
    monkeypatch.setattr(architecture_sweep.time, "perf_counter", fake_perf_counter)

    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("seed: 0\n", encoding="utf-8")
    output_path = tmp_path / "results.parquet"

    exit_code = architecture_sweep.main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--print-metrics",
        ]
    )
    assert exit_code == 0
    assert output_path.exists()

    frame = pd.read_parquet(output_path)
    obtained = []
    for record in frame.sort_values("label").to_dict(orient="records"):
        obtained.append({
            key: record[key]
            for key in (
                "label",
                "architecture",
                "model_dim",
                "depth",
                "horizon",
                "conv_dilations",
                "duration_seconds",
                "best_model_path",
                "metric_val_loss",
                "metric_train_loss",
                "dataset_samples",
                "dataset_batches",
            )
        })

    assert len(obtained) == len(expected_records)
    for obs, exp in zip(obtained, sorted(expected_records, key=lambda row: row["label"])):
        assert obs["label"] == exp["label"]
        assert obs["architecture"] == exp["architecture"]
        assert obs["model_dim"] == exp["model_dim"]
        assert obs["depth"] == exp["depth"]
        assert obs["horizon"] == exp["horizon"]
        assert obs["conv_dilations"] == exp["conv_dilations"]
        assert obs["best_model_path"] == exp["best_model_path"]
        assert pytest.approx(obs["duration_seconds"], rel=1e-6) == exp["duration_seconds"]
        assert pytest.approx(obs["metric_val_loss"], rel=1e-6) == exp["metric_val_loss"]
        assert pytest.approx(obs["metric_train_loss"], rel=1e-6) == exp["metric_train_loss"]
        assert obs["dataset_samples"] == exp["dataset_samples"]
        assert obs["dataset_batches"] == exp["dataset_batches"]
