import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


import numpy as np
import pandas as pd
import torch

from market_nn_plus_ultra.training import (
    DataConfig,
    ExperimentConfig,
    MaskedTimeSeriesLightningModule,
    ModelConfig,
    OptimizerConfig,
    PretrainingConfig,
    ReinforcementConfig,
    ReplayBufferConfig,
    TrainerConfig,
    run_reinforcement_finetuning,
)


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


def test_reinforcement_finetuning_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "ppo_fixture.db"
    _build_sqlite_fixture(db_path)

    data_config = DataConfig(
        sqlite_path=db_path,
        symbol_universe=["TEST"],
        feature_set=[],
        window_size=16,
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
        batch_size=4,
        num_workers=0,
        accelerator="cpu",
        precision="32-true",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    reinforcement_config = ReinforcementConfig(
        total_updates=1,
        steps_per_rollout=8,
        policy_epochs=1,
        minibatch_size=4,
        gamma=0.9,
        gae_lambda=0.8,
        learning_rate=1e-3,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
    )
    experiment = ExperimentConfig(
        seed=13,
        data=data_config,
        model=model_config,
        optimizer=OptimizerConfig(lr=1e-3),
        trainer=trainer_config,
        reinforcement=reinforcement_config,
    )

    result = run_reinforcement_finetuning(experiment, device="cpu")

    assert len(result.updates) == reinforcement_config.total_updates
    update = result.updates[0]
    assert np.isfinite(update.mean_reward)
    assert np.isfinite(update.policy_loss)
    assert np.isfinite(update.value_loss)
    assert np.isfinite(update.entropy)
    assert update.samples == reinforcement_config.steps_per_rollout
    assert result.policy_state_dict


def test_reinforcement_with_replay_buffer(tmp_path: Path) -> None:
    db_path = tmp_path / "ppo_fixture.db"
    _build_sqlite_fixture(db_path)

    data_config = DataConfig(
        sqlite_path=db_path,
        symbol_universe=["TEST"],
        feature_set=[],
        window_size=16,
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
        batch_size=4,
        num_workers=0,
        accelerator="cpu",
        precision="32-true",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    reinforcement_config = ReinforcementConfig(
        total_updates=2,
        steps_per_rollout=8,
        policy_epochs=1,
        minibatch_size=4,
        learning_rate=1e-3,
        replay_buffer=ReplayBufferConfig(enabled=True, capacity=32, sample_ratio=1.0, min_samples=8),
    )
    experiment = ExperimentConfig(
        seed=21,
        data=data_config,
        model=model_config,
        optimizer=OptimizerConfig(lr=1e-3),
        trainer=trainer_config,
        reinforcement=reinforcement_config,
    )

    result = run_reinforcement_finetuning(experiment, device="cpu")

    assert len(result.updates) == reinforcement_config.total_updates
    # First update uses only fresh rollout data; second should include replay samples.
    assert result.updates[0].samples == reinforcement_config.steps_per_rollout
    assert result.updates[1].samples > reinforcement_config.steps_per_rollout


def test_reinforcement_parallel_rollout_workers(tmp_path: Path) -> None:
    db_path = tmp_path / "ppo_fixture.db"
    _build_sqlite_fixture(db_path, rows=96)

    data_config = DataConfig(
        sqlite_path=db_path,
        symbol_universe=["TEST"],
        feature_set=[],
        window_size=16,
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
        batch_size=4,
        num_workers=0,
        accelerator="cpu",
        precision="32-true",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    reinforcement_config = ReinforcementConfig(
        total_updates=1,
        steps_per_rollout=8,
        policy_epochs=1,
        minibatch_size=4,
        learning_rate=1e-3,
        rollout_workers=2,
        worker_device="cpu",
    )
    experiment = ExperimentConfig(
        seed=17,
        data=data_config,
        model=model_config,
        optimizer=OptimizerConfig(lr=1e-3),
        trainer=trainer_config,
        reinforcement=reinforcement_config,
    )

    result = run_reinforcement_finetuning(experiment, device="cpu")

    assert len(result.updates) == reinforcement_config.total_updates
    update = result.updates[0]
    assert update.samples == reinforcement_config.steps_per_rollout * reinforcement_config.rollout_workers
    assert np.isfinite(update.mean_reward)


def test_reinforcement_warm_start_from_pretraining_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "ppo_fixture.db"
    _build_sqlite_fixture(db_path)

    data_config = DataConfig(
        sqlite_path=db_path,
        symbol_universe=["TEST"],
        feature_set=[],
        window_size=16,
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
        batch_size=4,
        num_workers=0,
        accelerator="cpu",
        precision="32-true",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    reinforcement_config = ReinforcementConfig(
        total_updates=0,
        steps_per_rollout=8,
        policy_epochs=1,
        minibatch_size=4,
    )
    pretraining_config = PretrainingConfig(objective="masked")
    optimizer_config = OptimizerConfig(lr=1e-3)
    experiment = ExperimentConfig(
        seed=7,
        data=data_config,
        model=model_config,
        optimizer=optimizer_config,
        trainer=trainer_config,
        reinforcement=reinforcement_config,
        pretraining=pretraining_config,
    )

    pretrain_module = MaskedTimeSeriesLightningModule(model_config, optimizer_config, pretraining_config)
    checkpoint_path = tmp_path / "pretrain.ckpt"
    torch.save({"state_dict": pretrain_module.state_dict()}, checkpoint_path)

    result = run_reinforcement_finetuning(
        experiment,
        pretrain_checkpoint_path=checkpoint_path,
        device="cpu",
    )

    assert result.updates == []
    policy_backbone = {
        key: tensor
        for key, tensor in result.policy_state_dict.items()
        if key.startswith("backbone.")
    }
    pretrain_backbone = {
        key: tensor
        for key, tensor in pretrain_module.state_dict().items()
        if key.startswith("backbone.")
    }
    assert policy_backbone
    assert pretrain_backbone
    for key, tensor in pretrain_backbone.items():
        torch.testing.assert_close(policy_backbone[key], tensor)
