"""Continuous retraining orchestrator for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Iterable, MutableMapping

import pandas as pd
import torch

from ..data.labelling import MarketRegimeLabellingConfig, generate_regime_labels
from ..data.sqlite_loader import SQLiteMarketDataset, SQLiteMarketSource
from ..data.validation import ValidationBundle, validate_sqlite_frames
from ..training import (
    ExperimentConfig,
    TrainingRunResult,
    load_experiment_from_file,
    run_pretraining,
    run_reinforcement_finetuning,
    run_training,
)
from ..training.train_loop import _normalise_logged_metrics


LOGGER = logging.getLogger(__name__)


SQLITE_TABLES: tuple[str, ...] = (
    "assets",
    "series",
    "indicators",
    "regimes",
    "trades",
    "benchmarks",
    "cross_asset_views",
)


@dataclass(slots=True)
class DatasetStageConfig:
    """Configuration describing dataset validation and mutation steps."""

    strict_validation: bool = True
    regenerate_regimes: bool = False
    regime_config: MarketRegimeLabellingConfig | None = None


class WarmStartStrategy(str):
    """Enumeration of supported warm-start strategies for PPO fine-tuning."""

    TRAINING = "training"
    PRETRAINING = "pretraining"
    NONE = "none"

    @classmethod
    def from_arg(cls, value: str | None) -> "WarmStartStrategy":
        if value is None:
            return cls.TRAINING
        normalised = value.lower()
        if normalised not in {cls.TRAINING, cls.PRETRAINING, cls.NONE}:
            raise ValueError(
                "warm start strategy must be one of 'training', 'pretraining', or 'none'"
            )
        return cls(normalised)


@dataclass(slots=True)
class RetrainingPlan:
    """End-to-end automation plan covering dataset, pretraining, training, PPO."""

    dataset_path: Path
    training_config: Path
    output_dir: Path
    dataset_stage: DatasetStageConfig = field(default_factory=DatasetStageConfig)
    pretraining_config: Path | None = None
    run_pretraining: bool = True
    run_training: bool = True
    reinforcement_config: Path | None = None
    run_reinforcement: bool = False
    warm_start: WarmStartStrategy = WarmStartStrategy.TRAINING


@dataclass(slots=True)
class RetrainingStageResult:
    """Result payload for a single automation stage."""

    name: str
    success: bool
    started_at: datetime
    completed_at: datetime
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


@dataclass(slots=True)
class RetrainingSummary:
    """Structured result of running :func:`run_retraining_plan`."""

    plan: RetrainingPlan
    stages: list[RetrainingStageResult]
    started_at: datetime
    completed_at: datetime

    def stage_artifacts(self, name: str) -> dict[str, Any]:
        for stage in self.stages:
            if stage.name == name:
                return stage.artifacts
        raise KeyError(f"Stage '{name}' not executed")


def _read_sqlite_tables(path: Path, tables: Iterable[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(path) as conn:
        for table in tables:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            exists = conn.execute(query, (table,)).fetchone()
            if not exists:
                continue
            frames[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    return frames


def _validate_dataset(plan: RetrainingPlan) -> tuple[RetrainingStageResult, ValidationBundle]:
    start = datetime.now(timezone.utc)
    LOGGER.info("Validating SQLite dataset at %s", plan.dataset_path)
    frames = _read_sqlite_tables(plan.dataset_path, SQLITE_TABLES)
    bundle = validate_sqlite_frames(frames)
    notes: list[str] = []
    artifacts: dict[str, Any] = {}

    if bundle.series is not None and plan.dataset_stage.strict_validation:
        dataset = SQLiteMarketDataset(
            SQLiteMarketSource(path=str(plan.dataset_path)),
            validate=True,
        )
        panel = dataset.load()
        artifacts["symbols"] = sorted(
            panel.index.get_level_values("symbol").unique().tolist()
        )
        artifacts["total_rows"] = int(len(panel))
        notes.append(
            f"validated panel with {artifacts['total_rows']} rows across {len(artifacts['symbols'])} symbols"
        )

    row_counts: MutableMapping[str, int] = {}
    for table in SQLITE_TABLES:
        frame = getattr(bundle, table if table != "series" else "series")
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            row_counts[table] = int(len(frame))
    if row_counts:
        artifacts["row_counts"] = dict(row_counts)

    completed = datetime.now(timezone.utc)
    result = RetrainingStageResult(
        name="dataset",
        success=True,
        started_at=start,
        completed_at=completed,
        artifacts=artifacts,
        notes=notes,
    )
    return result, bundle


def _regenerate_regimes(
    plan: RetrainingPlan,
    bundle: ValidationBundle,
) -> RetrainingStageResult:
    start = datetime.now(timezone.utc)
    LOGGER.info("Regenerating regime labels for %s", plan.dataset_path)
    if bundle.series is None:
        raise ValueError("Cannot regenerate regimes without a validated 'series' table")
    regimes = generate_regime_labels(
        bundle.series,
        config=plan.dataset_stage.regime_config,
        assets=bundle.assets,
    )
    with sqlite3.connect(plan.dataset_path) as conn:
        regimes.to_sql("regimes", conn, if_exists="replace", index=False)
    notes = [f"wrote {len(regimes)} regime rows"]
    completed = datetime.now(timezone.utc)
    return RetrainingStageResult(
        name="regimes",
        success=True,
        started_at=start,
        completed_at=completed,
        artifacts={"row_count": int(len(regimes))},
        notes=notes,
    )


def _prepare_config(
    config_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> ExperimentConfig:
    config = load_experiment_from_file(config_path)
    config.data = replace(config.data, sqlite_path=dataset_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.trainer = replace(config.trainer, checkpoint_dir=checkpoint_dir)
    return config


def _run_pretraining_stage(
    plan: RetrainingPlan,
    config_path: Path,
) -> tuple[RetrainingStageResult, str]:
    start = datetime.now(timezone.utc)
    output_dir = plan.output_dir / "pretraining"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _prepare_config(config_path, plan.dataset_path, output_dir)
    result = run_pretraining(config)
    best_model = result.get("best_model_path") or ""
    metrics = _normalise_logged_metrics(result.get("logged_metrics", {}))
    artifacts = {
        "best_checkpoint": best_model,
        "metrics": metrics,
    }
    completed = datetime.now(timezone.utc)
    stage_result = RetrainingStageResult(
        name="pretraining",
        success=True,
        started_at=start,
        completed_at=completed,
        artifacts=artifacts,
    )
    return stage_result, best_model


def _run_training_stage(
    plan: RetrainingPlan,
    config_path: Path,
) -> tuple[RetrainingStageResult, TrainingRunResult]:
    start = datetime.now(timezone.utc)
    output_dir = plan.output_dir / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _prepare_config(config_path, plan.dataset_path, output_dir)
    result = run_training(config)
    artifacts: dict[str, Any] = {
        "best_checkpoint": result.best_model_path,
        "metrics": result.logged_metrics,
        "dataset_summary": result.dataset_summary,
    }
    if result.profitability_summary:
        artifacts["profitability_summary"] = result.profitability_summary
    if result.profitability_reports:
        artifacts["profitability_reports"] = result.profitability_reports
    completed = datetime.now(timezone.utc)
    stage_result = RetrainingStageResult(
        name="training",
        success=True,
        started_at=start,
        completed_at=completed,
        artifacts=artifacts,
    )
    return stage_result, result


def _determine_reinforcement_checkpoint(
    warm_start: WarmStartStrategy,
    training_result: TrainingRunResult | None,
    pretraining_checkpoint: str | None,
) -> tuple[str | None, str | None]:
    if warm_start == WarmStartStrategy.NONE:
        return None, None
    if warm_start == WarmStartStrategy.TRAINING:
        if training_result is None or not training_result.best_model_path:
            raise ValueError("Training warm start requested but no training checkpoint available")
        return training_result.best_model_path, None
    if warm_start == WarmStartStrategy.PRETRAINING:
        if not pretraining_checkpoint:
            raise ValueError("Pretraining warm start requested but no pretraining checkpoint available")
        return None, pretraining_checkpoint
    raise ValueError(f"Unknown warm start strategy: {warm_start}")


def _run_reinforcement_stage(
    plan: RetrainingPlan,
    config_path: Path | None,
    training_result: TrainingRunResult | None,
    pretraining_checkpoint: str | None,
) -> RetrainingStageResult:
    start = datetime.now(timezone.utc)
    output_dir = plan.output_dir / "reinforcement"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = config_path or plan.training_config
    config = _prepare_config(base_config_path, plan.dataset_path, output_dir)

    checkpoint_path, pretrain_checkpoint = _determine_reinforcement_checkpoint(
        plan.warm_start,
        training_result,
        pretraining_checkpoint,
    )

    result = run_reinforcement_finetuning(
        config,
        checkpoint_path=checkpoint_path,
        pretrain_checkpoint_path=pretrain_checkpoint,
        device="cpu" if config.trainer.accelerator == "cpu" else "auto",
    )
    policy_path = output_dir / "policy_state_dict.pt"
    torch.save(result.policy_state_dict, policy_path)

    updates = result.updates
    artifacts: dict[str, Any] = {
        "policy_state_dict": str(policy_path),
        "updates": [update.__dict__ for update in updates],
    }
    if updates:
        artifacts["last_update"] = updates[-1].__dict__

    completed = datetime.now(timezone.utc)
    return RetrainingStageResult(
        name="reinforcement",
        success=True,
        started_at=start,
        completed_at=completed,
        artifacts=artifacts,
    )


def run_retraining_plan(plan: RetrainingPlan) -> RetrainingSummary:
    """Execute the requested stages and return a structured summary."""

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    stages: list[RetrainingStageResult] = []

    dataset_result, bundle = _validate_dataset(plan)
    stages.append(dataset_result)

    if plan.dataset_stage.regenerate_regimes:
        regime_stage = _regenerate_regimes(plan, bundle)
        stages.append(regime_stage)

    pretrain_checkpoint: str | None = None
    training_result: TrainingRunResult | None = None

    if plan.run_pretraining:
        if plan.pretraining_config is None:
            raise ValueError("run_pretraining=True requires 'pretraining_config'")
        pre_stage, checkpoint = _run_pretraining_stage(plan, plan.pretraining_config)
        stages.append(pre_stage)
        pretrain_checkpoint = checkpoint or None

    if plan.run_training:
        train_stage, train_result = _run_training_stage(plan, plan.training_config)
        stages.append(train_stage)
        training_result = train_result

    if plan.run_reinforcement:
        reinforcement_stage = _run_reinforcement_stage(
            plan,
            plan.reinforcement_config,
            training_result,
            pretrain_checkpoint,
        )
        stages.append(reinforcement_stage)

    completed_at = datetime.now(timezone.utc)
    return RetrainingSummary(
        plan=plan,
        stages=stages,
        started_at=started_at,
        completed_at=completed_at,
    )


__all__ = [
    "DatasetStageConfig",
    "WarmStartStrategy",
    "RetrainingPlan",
    "RetrainingStageResult",
    "RetrainingSummary",
    "run_retraining_plan",
]

