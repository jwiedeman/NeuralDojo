"""Experiment configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _expand(path: Path | str | None) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


@dataclass(slots=True)
class DataConfig:
    database_path: Path
    indicators: tuple[str, ...] = ()
    asset_universe: tuple[str, ...] = ()


@dataclass(slots=True)
class ModelConfig:
    input_size: Optional[int] = None
    d_model: int = 512
    depth: int = 16
    n_heads: int = 8
    patch_size: int = 1
    conv_kernel: int = 5
    conv_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.2
    ffn_expansion: int = 4
    forecast_horizon: int = 5
    output_size: int = 3


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000


@dataclass(slots=True)
class TrainingConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    batch_size: int = 256
    num_epochs: int = 100
    gradient_clip_val: float = 1.0
    mixed_precision: bool = True
    checkpoint_dir: Optional[Path] = None
    window_size: int = 256
    window_stride: int = 8
    target_column: str = "close"
    experiment_seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.data.database_path, (str, Path)):
            self.data.database_path = _expand(self.data.database_path)  # type: ignore[attr-defined]
        if isinstance(self.checkpoint_dir, (str, Path)):
            self.checkpoint_dir = _expand(self.checkpoint_dir)
