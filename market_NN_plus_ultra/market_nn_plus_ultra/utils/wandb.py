"""Weights & Biases helpers for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..training.config import ExperimentConfig


def _normalise_value(value: Any) -> Any:
    """Convert values into JSON-friendly representations."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise_value(item) for item in value]
    return value


def normalise_experiment_config(config: ExperimentConfig) -> dict[str, Any]:
    """Serialise an :class:`ExperimentConfig` for experiment tracking."""

    raw_dict = asdict(config)
    return _normalise_value(raw_dict)  # type: ignore[return-value]


def maybe_create_wandb_logger(
    config: ExperimentConfig,
    *,
    run_kind: str = "train",
):
    """Return a configured :class:`WandbLogger` when tracking is enabled."""

    if not config.wandb_project:
        return None

    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as exc:  # pragma: no cover - import guarded
        raise RuntimeError(
            "Weights & Biases logging requested but 'wandb' extras are not installed"
        ) from exc

    tags = list(config.wandb_tags)
    if run_kind and run_kind not in tags:
        tags.append(run_kind)

    wandb_config = normalise_experiment_config(config)
    wandb_config["run_kind"] = run_kind

    log_model = not config.wandb_offline
    logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        tags=tags or None,
        log_model=log_model,
        offline=config.wandb_offline,
        config=wandb_config,
    )
    return logger


__all__ = ["maybe_create_wandb_logger", "normalise_experiment_config"]
