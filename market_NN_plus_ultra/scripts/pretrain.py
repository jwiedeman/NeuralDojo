"""CLI entrypoint for running self-supervised pretraining."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from market_nn_plus_ultra.training import load_experiment_from_file, run_pretraining


def _maybe(value, cast):
    if value is None:
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return value


def _apply_overrides(config, args) -> None:
    if args.seed is not None:
        config.seed = args.seed

    trainer = config.trainer
    if args.devices is not None:
        trainer.devices = _maybe(args.devices, int)
    if args.accelerator is not None:
        trainer.accelerator = args.accelerator
    if args.max_epochs is not None:
        trainer.max_epochs = args.max_epochs
    if args.batch_size is not None:
        trainer.batch_size = args.batch_size

    model = config.model
    if args.architecture is not None:
        model.architecture = args.architecture.lower()
    if args.model_dim is not None:
        model.model_dim = args.model_dim
    if args.depth is not None:
        model.depth = args.depth
    if args.heads is not None:
        model.heads = args.heads
    if args.feature_dim is not None:
        model.feature_dim = args.feature_dim

    pretrain = config.pretraining
    if pretrain is not None:
        if args.mask_prob is not None:
            pretrain.mask_prob = args.mask_prob
        if args.mask_value is not None:
            try:
                pretrain.mask_value = float(args.mask_value)
            except ValueError:
                pretrain.mask_value = args.mask_value
        if args.pretrain_loss is not None:
            pretrain.loss = args.pretrain_loss
        if args.objective is not None:
            pretrain.objective = args.objective.lower()
        if args.temperature is not None:
            pretrain.temperature = args.temperature
        if args.projection_dim is not None:
            pretrain.projection_dim = args.projection_dim
        if args.augmentations is not None:
            items = [item.strip().lower() for item in args.augmentations.split(",") if item.strip()]
            pretrain.augmentations = tuple(items)
        if args.jitter_std is not None:
            pretrain.jitter_std = args.jitter_std
        if args.scaling_std is not None:
            pretrain.scaling_std = args.scaling_std
        if args.time_mask_ratio is not None:
            pretrain.time_mask_ratio = args.time_mask_ratio
        if args.time_mask_fill is not None:
            try:
                pretrain.time_mask_fill = float(args.time_mask_fill)
            except ValueError:
                pretrain.time_mask_fill = args.time_mask_fill

    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb_entity = args.wandb_entity
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.wandb_tags is not None:
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        config.wandb_tags = tuple(tags)
    if args.wandb_offline:
        config.wandb_offline = True
    if args.no_wandb:
        config.wandb_project = None
        config.wandb_run_name = None
        config.wandb_tags = ()


def _ensure_wandb_defaults(config, *, config_path: Path, disabled: bool) -> None:
    """Populate sensible Weights & Biases defaults unless explicitly disabled."""

    if disabled:
        return

    if not config.wandb_project:
        config.wandb_project = "plus-ultra"

    config_name = config_path.stem or "experiment"
    if not config.wandb_run_name:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        config.wandb_run_name = f"pretrain-{config_name}-{timestamp}"

    tags = list(config.wandb_tags)
    if config_name not in tags:
        tags.append(config_name)
    config.wandb_tags = tuple(dict.fromkeys(tags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run self-supervised pretraining (masked or contrastive) for Market NN Plus Ultra"
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--seed", type=int, help="Override the experiment seed")
    parser.add_argument("--devices", help="Override the trainer device setting (int or 'auto')")
    parser.add_argument("--accelerator", type=str, help="Override the trainer accelerator")
    parser.add_argument("--max-epochs", type=int, help="Override the maximum training epochs")
    parser.add_argument("--batch-size", type=int, help="Override the training batch size")
    parser.add_argument("--architecture", type=str, help="Select a different backbone architecture")
    parser.add_argument("--model-dim", type=int, help="Override the model hidden dimension")
    parser.add_argument("--depth", type=int, help="Override backbone depth")
    parser.add_argument("--heads", type=int, help="Override number of attention heads")
    parser.add_argument("--feature-dim", type=int, help="Override the expected feature dimension")
    parser.add_argument("--mask-prob", type=float, help="Override the masking probability")
    parser.add_argument(
        "--mask-value",
        type=str,
        help="Override the mask fill value (float or 'mean' to reuse window mean)",
    )
    parser.add_argument("--pretrain-loss", type=str, help="Override the pretraining loss type")
    parser.add_argument("--objective", type=str, help="Choose the pretraining objective (masked|contrastive)")
    parser.add_argument("--temperature", type=float, help="Override InfoNCE temperature for contrastive runs")
    parser.add_argument("--projection-dim", type=int, help="Set contrastive projection head dimension")
    parser.add_argument(
        "--augmentations",
        type=str,
        help="Comma separated list of contrastive augmentations (jitter,scaling,time_mask)",
    )
    parser.add_argument("--jitter-std", type=float, help="Std-dev for additive noise augmentation")
    parser.add_argument("--scaling-std", type=float, help="Std-dev for multiplicative scaling augmentation")
    parser.add_argument("--time-mask-ratio", type=float, help="Fraction of timesteps to mask in time_mask augmentation")
    parser.add_argument(
        "--time-mask-fill",
        type=str,
        help="Fill value for masked timesteps (float or 'mean')",
    )
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, help="Weights & Biases entity/username")
    parser.add_argument("--wandb-run-name", type=str, help="Explicit Weights & Biases run name")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        help="Comma separated list of tags to attach to the Weights & Biases run",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Force Weights & Biases into offline mode for air-gapped experiments",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable the automatic Weights & Biases run launched by the pretrainer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_from_file(args.config)
    if config.pretraining is None:
        raise SystemExit("Config must include a 'pretraining' section for self-supervised runs")
    _apply_overrides(config, args)
    _ensure_wandb_defaults(config, config_path=args.config, disabled=args.no_wandb)
    result = run_pretraining(config)
    print(f"Best checkpoint stored at: {result['best_model_path']}")


if __name__ == "__main__":
    main()
