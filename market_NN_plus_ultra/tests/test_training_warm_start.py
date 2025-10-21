import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_nn_plus_ultra.training import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainingConfig,
    TrainerConfig,
    run_training,
)
from market_nn_plus_ultra.training.pretrain_loop import MaskedTimeSeriesLightningModule


def _build_sqlite_fixture(path: Path, *, rows: int = 64) -> None:
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(rows)]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * rows,
            "open": np.linspace(100, 110, rows),
            "high": np.linspace(101, 111, rows),
            "low": np.linspace(99, 109, rows),
            "close": np.linspace(100.5, 110.5, rows),
            "volume": np.linspace(1_000, 2_000, rows),
        }
    )
    assets = pd.DataFrame(
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
        frame.to_sql("series", conn, index=False, if_exists="replace")
        assets.to_sql("assets", conn, index=False, if_exists="replace")


def test_run_training_warm_starts_from_pretraining_checkpoint(tmp_path, monkeypatch):
    db_path = tmp_path / "fixture.db"
    _build_sqlite_fixture(db_path)

    data_config = DataConfig(
        sqlite_path=db_path,
        symbol_universe=["TEST"],
        feature_set=[],
        window_size=16,
        horizon=4,
        stride=4,
        normalise=True,
        val_fraction=0.25,
    )
    model_config = ModelConfig(
        feature_dim=5,
        model_dim=32,
        depth=2,
        heads=4,
        dropout=0.1,
        conv_kernel_size=3,
        conv_dilations=(1, 2),
        horizon=4,
        output_dim=1,
        architecture="temporal_transformer",
    )
    trainer_config = TrainerConfig(
        batch_size=4,
        num_workers=0,
        persistent_workers=False,
        accelerator="cpu",
        precision="32-true",
        checkpoint_dir=tmp_path / "checkpoints",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    experiment = ExperimentConfig(
        seed=11,
        data=data_config,
        model=model_config,
        optimizer=OptimizerConfig(lr=1e-3),
        trainer=trainer_config,
        pretraining=PretrainingConfig(objective="masked"),
    )

    pretrain_module = MaskedTimeSeriesLightningModule(
        model_config,
        OptimizerConfig(lr=1e-3),
        experiment.pretraining,
    )
    checkpoint_path = tmp_path / "pretrain.ckpt"
    torch.save({"state_dict": pretrain_module.state_dict()}, checkpoint_path)

    trainer_instances = []

    class _StubTrainer:
        def __init__(self, *args, **kwargs):
            self.logged_metrics = {"val/loss": torch.tensor(0.5)}
            self.module_backbone = None
            trainer_instances.append(self)

        def fit(self, module, datamodule=None):
            self.module_backbone = {
                key: tensor.detach().cpu().clone()
                for key, tensor in module.backbone.state_dict().items()
            }

    class _StubModelCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = ""

    class _StubLearningRateMonitor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "market_nn_plus_ultra.training.train_loop.pl.Trainer",
        _StubTrainer,
    )
    monkeypatch.setattr(
        "market_nn_plus_ultra.training.train_loop.pl.callbacks.ModelCheckpoint",
        _StubModelCheckpoint,
    )
    monkeypatch.setattr(
        "market_nn_plus_ultra.training.train_loop.pl.callbacks.LearningRateMonitor",
        _StubLearningRateMonitor,
    )
    monkeypatch.setattr(
        "market_nn_plus_ultra.training.train_loop.maybe_create_wandb_logger",
        lambda *args, **kwargs: None,
    )

    result = run_training(experiment, pretrain_checkpoint_path=checkpoint_path)

    assert result.dataset_summary["train_windows"] > 0
    assert trainer_instances, "Expected stub trainer to be constructed"
    backbone_state = trainer_instances[-1].module_backbone
    assert backbone_state is not None

    pretrain_backbone = pretrain_module.backbone.state_dict()
    for key, tensor in pretrain_backbone.items():
        torch.testing.assert_close(backbone_state[key], tensor)

