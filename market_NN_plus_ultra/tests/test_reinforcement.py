import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from market_nn_plus_ultra.training import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    ReinforcementConfig,
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
    assert result.policy_state_dict
