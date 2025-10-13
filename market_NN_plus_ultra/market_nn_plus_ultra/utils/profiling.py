"""Throughput and memory profiling utilities for Plus Ultra backbones."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import torch

from ..training.config import ModelConfig, OptimizerConfig
from ..training.train_loop import MarketLightningModule


@dataclass(slots=True)
class ThroughputReport:
    """Summary of a single profiling run."""

    architecture: str
    device: str
    batch_size: int
    seq_len: int
    feature_dim: int
    seconds_per_step: float
    tokens_per_second: float
    peak_memory_mb: float | None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the report."""

        return {
            "architecture": self.architecture,
            "device": self.device,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "feature_dim": self.feature_dim,
            "seconds_per_step": self.seconds_per_step,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_mb": self.peak_memory_mb,
        }


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available")
    return resolved


def profile_backbone_throughput(
    model_config: ModelConfig,
    *,
    batch_size: int = 16,
    seq_len: int = 512,
    device: str | torch.device = "cpu",
    warmup_steps: int = 2,
    measure_steps: int = 5,
) -> ThroughputReport:
    """Profile a configured backbone and return throughput statistics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if measure_steps <= 0:
        raise ValueError("measure_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")

    device_obj = _resolve_device(device)
    module = MarketLightningModule(model_config, OptimizerConfig())
    module.eval()
    module.to(device_obj)

    features = torch.randn(batch_size, seq_len, model_config.feature_dim, device=device_obj)

    with torch.no_grad():
        for _ in range(warmup_steps):
            module.backbone(features)

    peak_memory_mb: float | None = None
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
        torch.cuda.reset_peak_memory_stats(device_obj)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_steps):
            module.backbone(features)
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
        peak_memory_mb = torch.cuda.max_memory_allocated(device_obj) / (1024**2)
    elapsed = time.perf_counter() - start

    seconds_per_step = elapsed / measure_steps
    tokens = batch_size * seq_len
    tokens_per_second = tokens / seconds_per_step if seconds_per_step > 0 else float("inf")

    return ThroughputReport(
        architecture=model_config.architecture,
        device=str(device_obj),
        batch_size=batch_size,
        seq_len=seq_len,
        feature_dim=model_config.feature_dim,
        seconds_per_step=seconds_per_step,
        tokens_per_second=tokens_per_second,
        peak_memory_mb=peak_memory_mb,
    )


__all__ = ["ThroughputReport", "profile_backbone_throughput"]
