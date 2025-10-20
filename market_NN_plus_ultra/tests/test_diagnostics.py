import types

import pytest
import torch
import pytorch_lightning as pl

from market_nn_plus_ultra.training import load_experiment_from_file
from market_nn_plus_ultra.training.diagnostics import (
    RunningMoments,
    TrainingDiagnosticsCallback,
)


class _DummyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(features.float())


def test_running_moments_statistics() -> None:
    stats = RunningMoments()
    stats.update(1.0)
    stats.update(3.0)
    assert stats.count == 2
    assert stats.mean == pytest.approx(2.0)
    assert stats.variance == pytest.approx(2.0)
    stats.reset()
    assert stats.count == 0
    assert stats.mean == 0.0


def test_training_diagnostics_callback_logs_metrics() -> None:
    module = _DummyModule()
    callback = TrainingDiagnosticsCallback(log_interval=1, profile=True)
    trainer = types.SimpleNamespace(
        logger=None,
        loggers=None,
        callback_metrics={},
        global_step=5,
    )
    for param in module.parameters():
        param.grad = torch.ones_like(param)
    callback.on_after_backward(trainer, module)
    assert "diagnostics/gradient_noise_ratio" in trainer.callback_metrics
    callback.on_validation_epoch_start(trainer, module)
    batch = {
        "features": torch.randn(8, 3),
        "targets": torch.randn(8, 1),
    }
    callback.on_validation_batch_end(trainer, module, None, batch, 0)
    callback.on_validation_epoch_end(trainer, module)
    assert "diagnostics/calibration_abs_error" in trainer.callback_metrics
    assert "diagnostics/calibration_bias" in trainer.callback_metrics


def test_experiment_loader_parses_diagnostics(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
seed: 7
notes: diagnostics

data:
  sqlite_path: data/fixture.db

model:
  feature_dim: 16

optimizer:
  lr: 0.001

trainer:
  batch_size: 4
  max_epochs: 1

diagnostics:
  enabled: true
  log_interval: 7
  profile: true
  gradient_noise_threshold: 3.0
  calibration_bias_threshold: 0.05
  calibration_error_threshold: 0.2
        """
    )
    config = load_experiment_from_file(config_path)
    diagnostics = config.diagnostics
    assert diagnostics.enabled is True
    assert diagnostics.log_interval == 7
    assert diagnostics.profile is True
    assert diagnostics.gradient_noise_threshold == pytest.approx(3.0)
    assert diagnostics.calibration_bias_threshold == pytest.approx(0.05)
    assert diagnostics.calibration_error_threshold == pytest.approx(0.2)
