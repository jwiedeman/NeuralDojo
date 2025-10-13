import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_nn_plus_ultra.training.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainingConfig,
    TrainerConfig,
)
from market_nn_plus_ultra.training.pretrain_loop import (
    ContrastiveTimeSeriesLightningModule,
    MaskedTimeSeriesLightningModule,
    instantiate_pretraining_module,
)


def _build_sqlite_fixture(path: Path, *, rows: int = 80) -> None:
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


def _base_configs(tmp_path: Path, objective: str) -> ExperimentConfig:
    db_path = tmp_path / "pretrain_fixture.db"
    _build_sqlite_fixture(db_path)

    data_config = DataConfig(
        sqlite_path=db_path,
        symbol_universe=["TEST"],
        feature_set=[],
        window_size=24,
        horizon=4,
        stride=4,
        normalise=True,
        val_fraction=0.0,
    )
    model_config = ModelConfig(
        feature_dim=5,
        model_dim=64,
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
        batch_size=8,
        num_workers=0,
        accelerator="cpu",
        precision="32-true",
        checkpoint_dir=tmp_path / "checkpoints",
        log_every_n_steps=1,
    )
    pretraining_config = PretrainingConfig(
        mask_prob=0.3,
        mask_value="mean",
        loss="mse",
        objective=objective,
        temperature=0.2,
        projection_dim=48,
        augmentations=("jitter", "time_mask"),
        jitter_std=0.01,
        scaling_std=0.05,
        time_mask_ratio=0.25,
        time_mask_fill="mean",
    )
    return ExperimentConfig(
        seed=11,
        data=data_config,
        model=model_config,
        optimizer=OptimizerConfig(lr=1e-3),
        trainer=trainer_config,
        pretraining=pretraining_config,
    )


def _next_batch(data_module):
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    return next(iter(loader))


def test_masked_pretraining_training_step(tmp_path: Path) -> None:
    config = _base_configs(tmp_path, objective="masked")
    module, data_module = instantiate_pretraining_module(config)
    assert isinstance(module, MaskedTimeSeriesLightningModule)
    batch = _next_batch(data_module)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_contrastive_pretraining_training_step(tmp_path: Path) -> None:
    config = _base_configs(tmp_path, objective="contrastive")
    module, data_module = instantiate_pretraining_module(config)
    assert isinstance(module, ContrastiveTimeSeriesLightningModule)
    batch = _next_batch(data_module)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    # InfoNCE loss should be non-negative
    assert float(loss.detach().cpu()) >= 0.0

