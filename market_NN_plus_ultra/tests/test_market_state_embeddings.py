from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from market_nn_plus_ultra.data.fixtures import FixtureConfig, build_fixture, write_fixture
from market_nn_plus_ultra.training.config import (
    CalibrationConfig,
    DataConfig,
    ExperimentConfig,
    MarketStateConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from market_nn_plus_ultra.training.train_loop import (
    MarketDataModule,
    ensure_feature_dim_alignment,
    instantiate_modules,
)


def _build_sqlite_fixture(path: Path, rows: int = 256) -> Path:
    config = FixtureConfig(
        symbols=["ALPHA", "BETA"],
        rows=rows,
        freq="1H",
        seed=11,
        start=datetime(2015, 1, 1),
        alt_features=2,
    )
    frames = build_fixture(config)
    return write_fixture(frames, path)


def test_data_module_emits_state_tokens(tmp_path: Path) -> None:
    db_path = _build_sqlite_fixture(tmp_path / "fixture.db")
    data_cfg = DataConfig(
        sqlite_path=db_path,
        window_size=64,
        horizon=4,
        stride=8,
        normalise=True,
        val_fraction=0.2,
    )
    trainer_cfg = TrainerConfig(batch_size=16, num_workers=0)
    data_module = MarketDataModule(data_cfg, trainer_cfg, seed=0)
    data_module.setup(stage="fit")

    assert data_module.market_state_feature_count > 0
    batch = next(iter(data_module.train_dataloader()))
    tokens = batch.get("state_tokens")
    assert tokens is not None
    assert tokens.dtype == torch.long
    assert tokens.shape[-1] == data_module.market_state_feature_count


def test_market_state_embeddings_extend_feature_dim(tmp_path: Path) -> None:
    db_path = _build_sqlite_fixture(tmp_path / "fixture.db", rows=320)
    data_cfg = DataConfig(
        sqlite_path=db_path,
        window_size=48,
        horizon=3,
        stride=6,
        normalise=True,
        val_fraction=0.2,
    )
    trainer_cfg = TrainerConfig(batch_size=8, num_workers=0)
    model_cfg = ModelConfig(
        feature_dim=4,
        model_dim=128,
        depth=4,
        heads=4,
        dropout=0.1,
        architecture="temporal_transformer",
        calibration=CalibrationConfig(enabled=False),
        market_state=MarketStateConfig(enabled=True, embedding_dim=8, dropout=0.0),
    )
    experiment = ExperimentConfig(
        seed=13,
        data=data_cfg,
        model=model_cfg,
        optimizer=OptimizerConfig(lr=1e-3),
        trainer=trainer_cfg,
    )

    data_module = MarketDataModule(experiment.data, experiment.trainer, seed=experiment.seed)
    ensure_feature_dim_alignment(experiment, data_module)
    module, data_module = instantiate_modules(experiment)
    if data_module.train_dataset is None:
        data_module.setup(stage="fit")

    base_dim = data_module.feature_dim
    state_count = data_module.market_state_feature_count
    assert state_count > 0
    expected_dim = base_dim + experiment.model.market_state.embedding_dim * state_count
    assert experiment.model.feature_dim == expected_dim
    assert module.market_state_embedding is not None
    assert module.market_state_embedding.output_dim == experiment.model.market_state.embedding_dim * state_count

    batch = next(iter(data_module.train_dataloader()))
    preds = module(batch["features"], state_tokens=batch.get("state_tokens"))
    assert preds.shape[0] == batch["features"].shape[0]
    assert preds.shape[-1] == experiment.model.output_dim
