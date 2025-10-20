from __future__ import annotations

from pathlib import Path

import torch
import yaml

from market_nn_plus_ultra.models.calibration import CalibratedPolicyHead, CalibrationHeadOutput
from market_nn_plus_ultra.training.config import CalibrationConfig, ModelConfig, OptimizerConfig
from market_nn_plus_ultra.training.train_loop import MarketLightningModule, load_experiment_from_file


def test_calibrated_policy_head_outputs_are_well_formed() -> None:
    head = CalibratedPolicyHead(
        model_dim=32,
        horizon=4,
        output_dim=2,
        quantile_levels=(0.1, 0.5, 0.9),
        dirichlet_temperature=0.75,
        min_concentration=0.02,
    )
    hidden = torch.randn(3, 16, 32)
    output = head(hidden)
    assert isinstance(output, CalibrationHeadOutput)
    assert output.prediction.shape == (3, 4, 2)
    assert output.quantiles.shape == (3, 4, 2, 3)
    # Quantiles should be monotonic when traversing along the quantile axis.
    diffs = torch.diff(output.quantiles, dim=-1)
    assert torch.all(diffs >= -1e-5)
    # Dirichlet concentration parameters must remain positive.
    assert torch.all(output.concentration > 0)


def test_lightning_module_tracks_latest_calibration_output() -> None:
    calibration = CalibrationConfig(
        enabled=True,
        quantiles=(0.2, 0.5, 0.8),
        temperature=0.8,
        min_concentration=0.05,
    )
    model_cfg = ModelConfig(
        feature_dim=8,
        model_dim=32,
        depth=1,
        heads=4,
        dropout=0.0,
        conv_kernel_size=3,
        conv_dilations=(1,),
        horizon=3,
        output_dim=2,
        architecture="temporal_transformer",
        max_seq_len=128,
        calibration=calibration,
    )
    module = MarketLightningModule(model_cfg, OptimizerConfig())
    batch_features = torch.randn(2, 12, 8)
    preds = module(batch_features)
    assert preds.shape == (2, 3, 2)
    head_output = module.latest_head_output
    assert head_output is not None
    assert head_output.quantiles.shape[-1] == len(calibration.quantiles)
    assert torch.all(head_output.concentration > 0)


def test_load_experiment_parses_calibration_block(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    raw = {
        "seed": 7,
        "data": {
            "sqlite_path": str(tmp_path / "data.db"),
            "target_columns": ["close"],
        },
        "model": {
            "feature_dim": 4,
            "model_dim": 16,
            "depth": 1,
            "heads": 2,
            "horizon": 2,
            "output_dim": 1,
            "architecture": "temporal_transformer",
            "calibration": {
                "enabled": True,
                "quantiles": [0.1, 0.6, 0.9],
                "temperature": 0.7,
                "min_concentration": 0.03,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(raw))
    experiment = load_experiment_from_file(config_path)
    calibration = experiment.model.calibration
    assert calibration.enabled is True
    assert calibration.quantiles == (0.1, 0.6, 0.9)
    assert calibration.temperature == 0.7
    assert calibration.min_concentration == 0.03

