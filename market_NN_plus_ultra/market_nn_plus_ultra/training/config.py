"""Experiment configuration dataclasses for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from ..trading.pnl import TradingCosts


@dataclass(slots=True)
class DataConfig:
    sqlite_path: Path
    symbol_universe: Optional[list[str]] = None
    indicators: dict[str, str] = field(default_factory=dict)
    resample_rule: Optional[str] = None
    tz_convert: Optional[str] = None
    feature_set: Optional[list[str]] = None
    target_columns: list[str] = field(default_factory=lambda: ["close"])
    window_size: int = 256
    horizon: int = 5
    stride: int = 1
    normalise: bool = True
    val_fraction: float = 0.2


@dataclass(slots=True)
class ModelConfig:
    feature_dim: int
    model_dim: int = 512
    depth: int = 16
    heads: int = 8
    dropout: float = 0.1
    conv_kernel_size: int = 5
    conv_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    horizon: int = 5
    output_dim: int = 3
    architecture: str = "hybrid_transformer"
    ff_mult: int = 4
    num_experts: int = 8
    router_dropout: float = 0.0
    ssm_state_dim: int = 256
    ssm_kernel_size: int = 16
    coarse_factor: int = 4
    cross_every: int = 2
    max_seq_len: int = 4096
    encoder_layers: Optional[int] = None
    decoder_layers: Optional[int] = None


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)


@dataclass(slots=True)
class TrainerConfig:
    batch_size: int = 32
    num_workers: int = 8
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    accelerator: str = "gpu"
    devices: Optional[int] = None
    precision: str = "bf16-mixed"
    log_every_n_steps: int = 50
    checkpoint_dir: Path = Path("checkpoints")
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"
    save_top_k: int = 1


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb_project: Optional[str] = None
    notes: Optional[str] = None
    pretraining: Optional["PretrainingConfig"] = None
    reinforcement: Optional["ReinforcementConfig"] = None


@dataclass(slots=True)
class PretrainingConfig:
    """Configuration for self-supervised pretraining tasks."""

    mask_prob: float = 0.25
    mask_value: float | str = 0.0
    loss: str = "mse"
    objective: str = "masked"
    temperature: float = 0.1
    projection_dim: int = 256
    augmentations: Tuple[str, ...] = ("jitter", "scaling", "time_mask")
    jitter_std: float = 0.02
    scaling_std: float = 0.1
    time_mask_ratio: float = 0.2
    time_mask_fill: float | str = 0.0
    monitor_metric: str = "val/pretrain_loss"


@dataclass(slots=True)
class ReinforcementConfig:
    """Hyper-parameters controlling policy-gradient fine-tuning."""

    total_updates: int = 50
    steps_per_rollout: int = 256
    policy_epochs: int = 4
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    targets_are_returns: bool = False
    activation: str = "tanh"
    costs: Optional[TradingCosts] = None

