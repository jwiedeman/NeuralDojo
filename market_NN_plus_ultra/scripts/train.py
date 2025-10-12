"""CLI entry-point for training Market NN Plus Ultra models."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from market_nn_plus_ultra.training.config import DataConfig, ModelConfig, OptimizerConfig, TrainingConfig
from market_nn_plus_ultra.training.train_loop import Trainer
from market_nn_plus_ultra.utils import set_seed


def _load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _to_tuple(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def build_training_config(raw: dict) -> TrainingConfig:
    experiment = raw.get("experiment", {})
    seed = experiment.get("seed", 42)

    data_cfg = DataConfig(
        database_path=raw["data"]["database_path"],
        indicators=_to_tuple(raw["data"].get("indicators", ())),
        asset_universe=_to_tuple(raw["data"].get("asset_universe", ())),
    )

    model_cfg = ModelConfig(
        input_size=raw["model"].get("input_size"),
        d_model=raw["model"].get("d_model", 512),
        depth=raw["model"].get("depth", 16),
        n_heads=raw["model"].get("n_heads", 8),
        patch_size=raw["model"].get("patch_size", 1),
        conv_kernel=raw["model"].get("conv_kernel", 5),
        conv_dilations=_to_tuple(raw["model"].get("conv_dilations", (1, 2, 4, 8))),
        dropout=raw["model"].get("dropout", 0.2),
        ffn_expansion=raw["model"].get("ffn_expansion", 4),
        forecast_horizon=raw["model"].get("forecast_horizon", 5),
        output_size=raw["model"].get("output_size", 3),
    )

    optimizer_cfg = OptimizerConfig(
        lr=raw["optimizer"].get("lr", 3e-4),
        weight_decay=raw["optimizer"].get("weight_decay", 1e-4),
        warmup_steps=raw["optimizer"].get("warmup_steps", 1000),
    )

    training_section = raw.get("training", {})
    training_cfg = TrainingConfig(
        data=data_cfg,
        model=model_cfg,
        optimizer=optimizer_cfg,
        batch_size=training_section.get("batch_size", 256),
        num_epochs=training_section.get("num_epochs", 100),
        gradient_clip_val=training_section.get("gradient_clip_val", 1.0),
        mixed_precision=training_section.get("mixed_precision", True),
        checkpoint_dir=training_section.get("checkpoint_dir"),
        window_size=training_section.get("window_size", 256),
        window_stride=training_section.get("window_stride", 8),
        target_column=training_section.get("target_column", "close"),
        experiment_seed=seed,
    )
    return training_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Market NN Plus Ultra trader")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    raw_cfg = _load_config(args.config)
    training_cfg = build_training_config(raw_cfg)

    seed = getattr(training_cfg, "experiment_seed", 42)
    set_seed(seed)

    trainer = Trainer(training_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
