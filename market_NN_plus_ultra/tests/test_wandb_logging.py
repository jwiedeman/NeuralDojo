from __future__ import annotations

from pathlib import Path

import pytest

from market_nn_plus_ultra.training import DataConfig, ExperimentConfig, ModelConfig
from market_nn_plus_ultra.utils.wandb import maybe_create_wandb_logger, normalise_experiment_config


def _make_config(tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig(
        seed=17,
        data=DataConfig(sqlite_path=tmp_path / "market.db"),
        model=ModelConfig(feature_dim=16),
    )
    config.trainer.checkpoint_dir = tmp_path / "ckpts"
    return config


def test_normalise_experiment_config_converts_paths(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    serialised = normalise_experiment_config(config)
    assert serialised["data"]["sqlite_path"] == str(tmp_path / "market.db")
    assert serialised["trainer"]["checkpoint_dir"] == str(tmp_path / "ckpts")


def test_maybe_create_wandb_logger_returns_none_when_disabled(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    logger = maybe_create_wandb_logger(config, run_kind="train")
    assert logger is None


def test_maybe_create_wandb_logger_creates_offline_run(tmp_path: Path) -> None:
    pytest.importorskip("wandb")
    from pytorch_lightning.loggers import WandbLogger

    config = _make_config(tmp_path)
    config.wandb_project = "plus-ultra-tests"
    config.wandb_tags = ("unit",)
    config.wandb_offline = True

    logger = maybe_create_wandb_logger(config, run_kind="train")
    assert isinstance(logger, WandbLogger)
    try:
        assert logger.experiment is not None
        assert "train" in logger.experiment.tags
        assert logger.experiment.config["run_kind"] == "train"
        assert logger._log_model is False  # offline mode should skip checkpoint uploads
    finally:
        logger.finalize("success")
