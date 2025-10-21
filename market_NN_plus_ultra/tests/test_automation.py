"""Tests for the continuous retraining orchestrator."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from market_nn_plus_ultra.automation import (
    DatasetStageConfig,
    RetrainingPlan,
    WarmStartStrategy,
    run_retraining_plan,
)
from market_nn_plus_ultra.training import TrainingRunResult
from market_nn_plus_ultra.training.reinforcement import ReinforcementRunResult


def _build_sqlite_fixture(path: Path, *, rows: int = 32) -> None:
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(rows)]
    series = pd.DataFrame(
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
        series.to_sql("series", conn, index=False, if_exists="replace")
        assets.to_sql("assets", conn, index=False, if_exists="replace")


def _write_experiment_config(path: Path, db_path: Path, *, include_pretraining: bool = False) -> Path:
    raw: dict[str, object] = {
        "seed": 7,
        "data": {
            "sqlite_path": str(db_path),
            "symbol_universe": ["TEST"],
            "indicators": {},
            "alternative_data": [],
            "resample_rule": None,
            "tz_convert": None,
            "feature_set": [],
            "target_columns": ["close"],
            "window_size": 8,
            "horizon": 2,
            "stride": 1,
            "normalise": True,
            "val_fraction": 0.25,
        },
        "model": {
            "feature_dim": 5,
            "model_dim": 16,
            "depth": 1,
            "heads": 2,
            "dropout": 0.1,
            "conv_kernel_size": 3,
            "conv_dilations": [1, 2],
            "horizon": 2,
            "output_dim": 1,
            "architecture": "temporal_transformer",
        },
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.99]},
        "trainer": {
            "batch_size": 4,
            "num_workers": 0,
            "persistent_workers": False,
            "max_epochs": 1,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "matmul_precision": None,
            "log_every_n_steps": 1,
            "checkpoint_dir": str(path.parent / "ckpts"),
            "monitor_metric": "val/loss",
            "monitor_mode": "min",
            "save_top_k": 1,
            "num_sanity_val_steps": 0,
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
        },
        "diagnostics": {
            "enabled": False,
            "log_interval": 10,
            "profile": False,
            "gradient_noise_threshold": None,
            "calibration_bias_threshold": None,
            "calibration_error_threshold": None,
        },
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_run_name": None,
        "wandb_tags": [],
        "wandb_offline": True,
    }
    if include_pretraining:
        raw["pretraining"] = {
            "mask_prob": 0.2,
            "mask_value": 0.0,
            "loss": "mse",
            "objective": "masked",
            "temperature": 0.1,
            "projection_dim": 8,
            "augmentations": ["jitter"],
            "jitter_std": 0.01,
            "scaling_std": 0.05,
            "time_mask_ratio": 0.1,
            "time_mask_fill": 0.0,
            "monitor_metric": "val/pretrain_loss",
        }
    path.write_text(yaml.safe_dump(raw))
    return path


def test_run_retraining_plan_executes_all_stages(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.db"
    _build_sqlite_fixture(db_path)
    train_config = _write_experiment_config(tmp_path / "train.yaml", db_path)
    pretrain_config = _write_experiment_config(
        tmp_path / "pretrain.yaml", db_path, include_pretraining=True
    )

    calls: list[tuple[str, object]] = []

    def fake_pretraining(config):
        calls.append(("pretrain", config.data.sqlite_path, config.trainer.checkpoint_dir))
        return {"best_model_path": str(tmp_path / "pre.ckpt"), "logged_metrics": {"val": torch.tensor(1.0)}}

    def fake_training(config):
        calls.append(("train", config.data.sqlite_path, config.trainer.checkpoint_dir))
        return TrainingRunResult(
            best_model_path=str(tmp_path / "train.ckpt"),
            logged_metrics={"val/loss": 0.42},
            dataset_summary={
                "train_windows": 8,
                "val_windows": 4,
                "train_batches": 2,
                "val_batches": 1,
                "feature_dim": 5,
                "market_state_features": 0,
            },
        )

    def fake_reinforcement(config, *, checkpoint_path=None, pretrain_checkpoint_path=None, device="cpu"):
        calls.append(("reinforce", checkpoint_path, pretrain_checkpoint_path, device))
        return ReinforcementRunResult(
            updates=[],
            policy_state_dict={"weight": torch.tensor(1.0)},
            evaluation_metrics={},
        )

    monkeypatch.setattr(
        "market_nn_plus_ultra.automation.retraining.run_pretraining",
        fake_pretraining,
    )
    monkeypatch.setattr(
        "market_nn_plus_ultra.automation.retraining.run_training",
        fake_training,
    )
    monkeypatch.setattr(
        "market_nn_plus_ultra.automation.retraining.run_reinforcement_finetuning",
        fake_reinforcement,
    )

    plan = RetrainingPlan(
        dataset_path=db_path,
        training_config=train_config,
        output_dir=tmp_path / "orchestration",
        dataset_stage=DatasetStageConfig(strict_validation=False),
        pretraining_config=pretrain_config,
        run_pretraining=True,
        run_training=True,
        run_reinforcement=True,
        warm_start=WarmStartStrategy.TRAINING,
    )

    summary = run_retraining_plan(plan)

    stage_names = [stage.name for stage in summary.stages]
    assert stage_names == ["dataset", "pretraining", "training", "reinforcement"]
    assert any(stage.artifacts for stage in summary.stages)

    assert calls[0][0] == "pretrain"
    assert calls[1][0] == "train"
    assert calls[2][0] == "reinforce"
    # Training warm start should pass the supervised checkpoint
    _, checkpoint, _, _ = calls[2]
    assert checkpoint == str(tmp_path / "train.ckpt")
    policy_path = summary.stage_artifacts("reinforcement")["policy_state_dict"]
    assert Path(policy_path).exists()


def test_pretraining_warm_start(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.db"
    _build_sqlite_fixture(db_path)
    pretrain_config = _write_experiment_config(
        tmp_path / "pretrain.yaml", db_path, include_pretraining=True
    )

    def fake_pretraining(config):
        return {"best_model_path": str(tmp_path / "pre.ckpt"), "logged_metrics": {}}

    def fake_reinforcement(config, *, checkpoint_path=None, pretrain_checkpoint_path=None, device="cpu"):
        assert checkpoint_path is None
        assert pretrain_checkpoint_path == str(tmp_path / "pre.ckpt")
        return ReinforcementRunResult(updates=[], policy_state_dict={}, evaluation_metrics={})

    monkeypatch.setattr(
        "market_nn_plus_ultra.automation.retraining.run_pretraining",
        fake_pretraining,
    )
    monkeypatch.setattr(
        "market_nn_plus_ultra.automation.retraining.run_reinforcement_finetuning",
        fake_reinforcement,
    )

    plan = RetrainingPlan(
        dataset_path=db_path,
        training_config=pretrain_config,
        output_dir=tmp_path / "orchestration",
        dataset_stage=DatasetStageConfig(strict_validation=False),
        pretraining_config=pretrain_config,
        run_pretraining=True,
        run_training=False,
        run_reinforcement=True,
        warm_start=WarmStartStrategy.PRETRAINING,
    )

    summary = run_retraining_plan(plan)
    assert [stage.name for stage in summary.stages] == ["dataset", "pretraining", "reinforcement"]


def test_missing_pretraining_checkpoint_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.db"
    _build_sqlite_fixture(db_path)
    train_config = _write_experiment_config(tmp_path / "train.yaml", db_path)

    plan = RetrainingPlan(
        dataset_path=db_path,
        training_config=train_config,
        output_dir=tmp_path / "orchestration",
        dataset_stage=DatasetStageConfig(strict_validation=False),
        pretraining_config=None,
        run_pretraining=False,
        run_training=False,
        run_reinforcement=True,
        warm_start=WarmStartStrategy.PRETRAINING,
    )

    try:
        run_retraining_plan(plan)
    except ValueError as exc:
        assert "pretraining" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected ValueError when warm starting without checkpoint")

