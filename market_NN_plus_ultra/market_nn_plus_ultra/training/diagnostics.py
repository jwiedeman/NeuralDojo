"""Training telemetry callbacks for gradient noise and calibration drift."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

_EPS = 1e-8


class RunningMoments:
    """Maintain running mean and variance via Welford's algorithm."""

    __slots__ = ("_count", "_mean", "_m2")

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean if self._count else 0.0

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass(slots=True)
class DiagnosticsThresholds:
    """Thresholds that trigger warnings when exceeded."""

    gradient_noise: float | None = None
    calibration_bias: float | None = None
    calibration_error: float | None = None


class TrainingDiagnosticsCallback(pl.Callback):
    """Estimate gradient noise and calibration drift during training."""

    def __init__(
        self,
        *,
        log_interval: int = 50,
        profile: bool = False,
        thresholds: DiagnosticsThresholds | None = None,
    ) -> None:
        super().__init__()
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")
        self.log_interval = int(log_interval)
        self.profile = profile
        self.thresholds = thresholds or DiagnosticsThresholds()
        self._step = 0
        self._grad_stats = RunningMoments()
        self._calibration_abs = RunningMoments()
        self._calibration_bias = RunningMoments()
        self._calibration_spread = RunningMoments()
        self._latest_gradient_ratio: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._step += 1
        total_square_norm = 0.0
        param_count = 0
        for param in pl_module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            grad_float = grad.float()
            total_square_norm += grad_float.pow(2).sum().item()
            param_count += grad.numel()

        if param_count == 0 or total_square_norm <= 0.0:
            return

        global_norm = math.sqrt(total_square_norm)
        self._grad_stats.update(global_norm)

        if self._step % self.log_interval == 0:
            ratio = 0.0
            if self._grad_stats.count > 1 and self._grad_stats.mean > 0.0:
                ratio = self._grad_stats.variance / (self._grad_stats.mean ** 2 + _EPS)
            self._latest_gradient_ratio = ratio
            metrics: Dict[str, float] = {
                "diagnostics/gradient_norm": global_norm,
                "diagnostics/gradient_noise_ratio": ratio,
            }
            if self.profile:
                metrics["diagnostics/gradient_norm_mean"] = self._grad_stats.mean
                metrics["diagnostics/gradient_norm_std"] = self._grad_stats.std
            self._log_metrics(trainer, metrics)
            if (
                self.thresholds.gradient_noise is not None
                and ratio > self.thresholds.gradient_noise
            ):
                logger.warning(
                    "Gradient noise ratio %.3f exceeded threshold %.3f", ratio, self.thresholds.gradient_noise
                )

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401
        """Reset calibration trackers at the start of validation."""

        self._calibration_abs.reset()
        self._calibration_bias.reset()
        self._calibration_spread.reset()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch_idx, dataloader_idx
        if not isinstance(batch, dict):
            return
        features = batch.get("features")
        targets = batch.get("targets")
        if features is None or targets is None:
            return
        with torch.no_grad():
            preds = pl_module(features)
        preds = preds.detach()
        targets = torch.as_tensor(targets).to(preds.device)
        if preds.shape != targets.shape:
            try:
                targets = targets.expand_as(preds)
            except RuntimeError:
                return
        diff = (preds - targets).float()
        if diff.numel() == 0:
            return
        abs_error = diff.abs().mean().item()
        bias = diff.mean().item()
        spread = diff.std(unbiased=False).item()
        self._calibration_abs.update(abs_error)
        self._calibration_bias.update(bias)
        self._calibration_spread.update(spread)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._calibration_abs.count == 0:
            return
        metrics: Dict[str, float] = {
            "diagnostics/calibration_abs_error": self._calibration_abs.mean,
            "diagnostics/calibration_bias": self._calibration_bias.mean,
        }
        if self.profile:
            metrics["diagnostics/calibration_spread"] = self._calibration_spread.mean
            metrics["diagnostics/calibration_bias_std"] = self._calibration_bias.std
        self._log_metrics(trainer, metrics)

        if (
            self.thresholds.calibration_error is not None
            and metrics["diagnostics/calibration_abs_error"] > self.thresholds.calibration_error
        ):
            logger.warning(
                "Calibration absolute error %.3f exceeded threshold %.3f",
                metrics["diagnostics/calibration_abs_error"],
                self.thresholds.calibration_error,
            )
        if (
            self.thresholds.calibration_bias is not None
            and abs(metrics["diagnostics/calibration_bias"]) > self.thresholds.calibration_bias
        ):
            logger.warning(
                "Calibration bias %.3f exceeded threshold ±%.3f",
                metrics["diagnostics/calibration_bias"],
                self.thresholds.calibration_bias,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_metrics(self, trainer: pl.Trainer, metrics: Dict[str, float]) -> None:
        safe_metrics = {key: float(value) for key, value in metrics.items()}
        callback_metrics = getattr(trainer, "callback_metrics", None)
        if isinstance(callback_metrics, dict):
            for key, value in safe_metrics.items():
                callback_metrics[key] = torch.tensor(value)
        for logger_instance in self._iter_loggers(trainer):
            log_fn = getattr(logger_instance, "log_metrics", None)
            if callable(log_fn):
                log_fn(dict(safe_metrics), step=getattr(trainer, "global_step", 0))

    def _iter_loggers(self, trainer: pl.Trainer) -> Iterable:
        logger_instance = getattr(trainer, "logger", None)
        if logger_instance not in (None, True):
            yield logger_instance
        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            return
        if isinstance(loggers, (list, tuple)):
            for entry in loggers:
                if entry not in (None, True):
                    yield entry
        elif loggers not in (None, True):
            yield loggers


__all__ = [
    "DiagnosticsThresholds",
    "RunningMoments",
    "TrainingDiagnosticsCallback",
]
