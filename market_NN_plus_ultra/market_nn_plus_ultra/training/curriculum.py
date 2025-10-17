"""Curriculum utilities for progressively widening temporal context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl

from .config import CurriculumConfig, CurriculumStage, DataConfig


@dataclass(slots=True, frozen=True)
class CurriculumParameters:
    """Resolved curriculum parameters for a training epoch."""

    window_size: int
    horizon: int
    stride: int
    normalise: bool


class CurriculumScheduler:
    """Decide which curriculum stage applies at a given epoch."""

    def __init__(self, base_config: DataConfig, curriculum: CurriculumConfig) -> None:
        if not curriculum.stages:
            raise ValueError("CurriculumConfig.stages must contain at least one stage")

        stages = sorted(curriculum.stages, key=lambda s: s.start_epoch)
        if stages[0].start_epoch > 0:
            stages.insert(0, CurriculumStage(start_epoch=0))

        for stage in stages:
            if stage.start_epoch < 0:
                raise ValueError("Curriculum stage start_epoch must be non-negative")

        self._base = CurriculumParameters(
            window_size=base_config.window_size,
            horizon=base_config.horizon,
            stride=base_config.stride,
            normalise=base_config.normalise,
        )
        self._stages = stages
        self._repeat_final = curriculum.repeat_final

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")

        last_stage: Optional[CurriculumStage] = None
        for stage in self._stages:
            if epoch < stage.start_epoch:
                break
            last_stage = stage

        if last_stage is not None:
            return last_stage

        return self._stages[0]

    def parameters_for_epoch(self, epoch: int) -> CurriculumParameters:
        if not self._repeat_final and epoch > self._stages[-1].start_epoch:
            raise ValueError(
                "Epoch exceeds final curriculum stage and repeat_final is disabled"
            )

        if epoch > self._stages[-1].start_epoch and self._repeat_final:
            stage = self._stages[-1]
        else:
            stage = self.stage_for_epoch(epoch)

        return CurriculumParameters(
            window_size=stage.window_size or self._base.window_size,
            horizon=stage.horizon or self._base.horizon,
            stride=stage.stride or self._base.stride,
            normalise=stage.normalise if stage.normalise is not None else self._base.normalise,
        )


class CurriculumCallback(pl.Callback):
    """Lightning callback that applies curriculum updates to the data module."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "step_curriculum"):
            return

        params = datamodule.step_curriculum(trainer.current_epoch)  # type: ignore[attr-defined]
        if params is None:
            return

        if hasattr(pl_module, "update_horizon"):
            pl_module.update_horizon(params.horizon)  # type: ignore[attr-defined]

        # Ensure dataloaders pick up the newly constructed datasets for the epoch.
        reset_train = getattr(trainer, "reset_train_dataloader", None)
        if callable(reset_train):
            reset_train()
        reset_val = getattr(trainer, "reset_val_dataloader", None)
        if callable(reset_val):
            reset_val()

        if hasattr(pl_module, "log"):
            pl_module.log("curriculum/window_size", float(params.window_size), prog_bar=True, logger=True)
            pl_module.log("curriculum/horizon", float(params.horizon), prog_bar=True, logger=True)

