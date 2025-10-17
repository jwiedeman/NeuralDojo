from pathlib import Path

import pytest
import torch

from market_nn_plus_ultra.training import (
    BenchmarkScenario,
    TrainerOverrides,
    flatten_benchmark_result,
    prepare_config_for_scenario,
)
from market_nn_plus_ultra.training.config import DataConfig, ExperimentConfig, ModelConfig
from market_nn_plus_ultra.training.train_loop import TrainingRunResult, _normalise_logged_metrics


@pytest.fixture
def base_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=7,
        data=DataConfig(sqlite_path=tmp_path / "fixture.db"),
        model=ModelConfig(feature_dim=8),
    )


def test_prepare_config_for_scenario_applies_overrides(base_config: ExperimentConfig) -> None:
    scenario = BenchmarkScenario(
        architecture="omni",
        model_dim=1024,
        depth=12,
        horizon=15,
        label="omni-large",
    )
    overrides = TrainerOverrides(max_epochs=2, batch_size=64)
    config = prepare_config_for_scenario(base_config, scenario, overrides=overrides)

    assert config is not base_config
    assert config.model.architecture == "omni"
    assert config.model.model_dim == 1024
    assert config.model.depth == 12
    assert config.model.horizon == 15
    assert config.data.horizon == 15
    assert config.trainer.max_epochs == overrides.max_epochs
    assert config.trainer.batch_size == overrides.batch_size
    assert config.wandb_offline is True
    # Ensure the base config remains unchanged
    assert base_config.model.architecture == "hybrid_transformer"
    assert base_config.data.horizon == 5


def test_flatten_benchmark_result_sanitises_keys() -> None:
    scenario = BenchmarkScenario("hybrid_transformer", 512, 8, 5)
    result = TrainingRunResult(
        best_model_path="checkpoints/model.ckpt",
        logged_metrics={"val/loss": 0.1234, "train/loss": 0.5678},
        dataset_summary={"train_windows": 100, "val_windows": 20},
    )
    row = flatten_benchmark_result(scenario, result, duration_seconds=3.5)
    assert row["metric_val_loss"] == pytest.approx(0.1234)
    assert row["metric_train_loss"] == pytest.approx(0.5678)
    assert row["dataset_train_windows"] == 100
    assert row["duration_seconds"] == pytest.approx(3.5)


def test_normalise_logged_metrics_handles_tensors() -> None:
    metrics = {"val/loss": torch.tensor(0.42), "accuracy": torch.tensor(0.9)}
    normalised = _normalise_logged_metrics(metrics)
    assert normalised["val/loss"] == pytest.approx(0.42)
    assert normalised["accuracy"] == pytest.approx(0.9)
