"""Helpers for orchestrating model architecture benchmark sweeps."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterator, Sequence

from .config import ExperimentConfig, TrainerConfig
from .train_loop import TrainingRunResult


@dataclass(slots=True)
class BenchmarkScenario:
    """Represents one combination of model hyper-parameters to benchmark."""

    architecture: str
    model_dim: int
    depth: int
    horizon: int
    label: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "architecture": self.architecture,
            "model_dim": self.model_dim,
            "depth": self.depth,
            "horizon": self.horizon,
        }
        if self.label is not None:
            data["label"] = self.label
        return data


@dataclass(slots=True)
class TrainerOverrides:
    """Lightweight container for trainer-level overrides used during sweeps."""

    max_epochs: int | None = None
    limit_train_batches: float | int | None = None
    limit_val_batches: float | int | None = None
    batch_size: int | None = None
    accelerator: str | None = None
    devices: int | str | None = None
    log_every_n_steps: int | None = None

    def apply(self, trainer: TrainerConfig) -> None:
        if self.max_epochs is not None:
            trainer.max_epochs = self.max_epochs
        if self.limit_train_batches is not None:
            trainer.limit_train_batches = self.limit_train_batches
        if self.limit_val_batches is not None:
            trainer.limit_val_batches = self.limit_val_batches
        if self.batch_size is not None:
            trainer.batch_size = self.batch_size
        if self.accelerator is not None:
            trainer.accelerator = self.accelerator
        if self.devices is not None:
            trainer.devices = self.devices  # type: ignore[assignment]
        if self.log_every_n_steps is not None:
            trainer.log_every_n_steps = self.log_every_n_steps

    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.max_epochs,
                self.limit_train_batches,
                self.limit_val_batches,
                self.batch_size,
                self.accelerator,
                self.devices,
                self.log_every_n_steps,
            )
        )


def prepare_config_for_scenario(
    base_config: ExperimentConfig,
    scenario: BenchmarkScenario,
    *,
    overrides: TrainerOverrides | None = None,
    disable_wandb: bool = True,
) -> ExperimentConfig:
    """Return a deep-copied config tuned for the provided benchmark scenario."""

    config = copy.deepcopy(base_config)
    model = config.model
    model.architecture = scenario.architecture
    model.model_dim = scenario.model_dim
    model.depth = scenario.depth
    model.horizon = scenario.horizon

    data = config.data
    data.horizon = scenario.horizon

    if disable_wandb:
        config.wandb_project = None
        config.wandb_run_name = None
        config.wandb_entity = None
        config.wandb_tags = ()
        config.wandb_offline = True

    if overrides is not None:
        overrides.apply(config.trainer)

    return config


def iter_scenarios(
    architectures: Sequence[str],
    model_dims: Sequence[int],
    depths: Sequence[int],
    horizons: Sequence[int],
    *,
    label_template: str | None = None,
) -> Iterator[BenchmarkScenario]:
    """Cartesian product helper used to enumerate benchmark scenarios."""

    for arch in architectures:
        for model_dim in model_dims:
            for depth in depths:
                for horizon in horizons:
                    label = None
                    if label_template is not None:
                        label = label_template.format(
                            architecture=arch,
                            model_dim=model_dim,
                            depth=depth,
                            horizon=horizon,
                        )
                    yield BenchmarkScenario(
                        architecture=arch,
                        model_dim=model_dim,
                        depth=depth,
                        horizon=horizon,
                        label=label,
                    )


def flatten_benchmark_result(
    scenario: BenchmarkScenario,
    result: TrainingRunResult,
    *,
    duration_seconds: float,
) -> dict[str, object]:
    """Convert a benchmark outcome into a row suitable for tabular storage."""

    row = scenario.to_dict()
    row.update(
        {
            "duration_seconds": duration_seconds,
            "best_model_path": result.best_model_path,
        }
    )
    for key, value in result.logged_metrics.items():
        safe_key = key.replace("/", "_")
        row[f"metric_{safe_key}"] = value
    for key, value in result.dataset_summary.items():
        row[f"dataset_{key}"] = value
    return row


__all__ = [
    "BenchmarkScenario",
    "TrainerOverrides",
    "prepare_config_for_scenario",
    "iter_scenarios",
    "flatten_benchmark_result",
]
