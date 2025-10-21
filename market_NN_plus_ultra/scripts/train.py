"""CLI entrypoint for training Market NN Plus Ultra models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from market_nn_plus_ultra.training import load_experiment_from_file, run_training


def _maybe(value, cast):
    if value is None:
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return value


def _apply_overrides(config, args) -> None:
    """Mutate the experiment config in-place based on CLI overrides."""

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
    if args.horizon is not None:
        model.horizon = args.horizon
    if args.feature_dim is not None:
        model.feature_dim = args.feature_dim

    optimizer = config.optimizer
    if args.learning_rate is not None:
        optimizer.lr = args.learning_rate
    if args.weight_decay is not None:
        optimizer.weight_decay = args.weight_decay
    if args.beta1 is not None or args.beta2 is not None:
        beta1, beta2 = optimizer.betas
        if args.beta1 is not None:
            beta1 = args.beta1
        if args.beta2 is not None:
            beta2 = args.beta2
        optimizer.betas = (beta1, beta2)

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

    diagnostics = config.diagnostics
    if args.enable_diagnostics:
        diagnostics.enabled = True
    if args.disable_diagnostics:
        diagnostics.enabled = False
    if args.diagnostics_interval is not None:
        diagnostics.enabled = True
        diagnostics.log_interval = max(1, args.diagnostics_interval)
    if args.diagnostics_profile:
        diagnostics.enabled = True
        diagnostics.profile = True
    if args.diagnostics_noise_threshold is not None:
        diagnostics.gradient_noise_threshold = args.diagnostics_noise_threshold
    if args.diagnostics_bias_threshold is not None:
        diagnostics.calibration_bias_threshold = args.diagnostics_bias_threshold
    if args.diagnostics_error_threshold is not None:
        diagnostics.calibration_error_threshold = args.diagnostics_error_threshold


def _ensure_wandb_defaults(config, *, config_path: Path, disabled: bool) -> None:
    """Populate sensible Weights & Biases defaults unless explicitly disabled."""

    if disabled:
        return

    if not config.wandb_project:
        config.wandb_project = "plus-ultra"

    config_name = config_path.stem or "experiment"
    if not config.wandb_run_name:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        config.wandb_run_name = f"train-{config_name}-{timestamp}"

    tags = list(config.wandb_tags)
    if config_name not in tags:
        tags.append(config_name)
    config.wandb_tags = tuple(dict.fromkeys(tags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Market NN Plus Ultra trading agent")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Experiment YAML path")
    parser.add_argument("--seed", type=int, help="Override the experiment seed")
    parser.add_argument("--devices", help="Override the trainer device setting (int or 'auto')")
    parser.add_argument("--accelerator", type=str, help="Override the trainer accelerator (cpu/gpu/mps)")
    parser.add_argument("--max-epochs", type=int, help="Override the max epoch count")
    parser.add_argument("--batch-size", type=int, help="Override the training batch size")
    parser.add_argument("--architecture", type=str, help="Choose a different model architecture")
    parser.add_argument("--model-dim", type=int, help="Override the model hidden dimension")
    parser.add_argument("--depth", type=int, help="Override the backbone depth")
    parser.add_argument("--heads", type=int, help="Override the attention head count")
    parser.add_argument("--horizon", type=int, help="Override the forecast horizon")
    parser.add_argument("--feature-dim", type=int, help="Override the expected feature dimension")
    parser.add_argument("--learning-rate", type=float, help="Override the optimiser learning rate")
    parser.add_argument("--weight-decay", type=float, help="Override the optimiser weight decay")
    parser.add_argument("--beta1", type=float, help="Override the first Adam beta coefficient")
    parser.add_argument("--beta2", type=float, help="Override the second Adam beta coefficient")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, help="Weights & Biases entity/username")
    parser.add_argument("--wandb-run-name", type=str, help="Explicit run name for Weights & Biases")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        help="Comma separated list of tags to attach to the Weights & Biases run",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Force Weights & Biases into offline mode for air-gapped environments",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable the automatic Weights & Biases run launched by the trainer",
    )
    parser.add_argument(
        "--enable-diagnostics",
        action="store_true",
        help="Force-enable training diagnostics regardless of the config default",
    )
    parser.add_argument(
        "--disable-diagnostics",
        action="store_true",
        help="Disable training diagnostics even if they are enabled in the config",
    )
    parser.add_argument(
        "--diagnostics-interval",
        type=int,
        help="Log gradient diagnostics every N optimisation steps",
    )
    parser.add_argument(
        "--diagnostics-profile",
        action="store_true",
        help="Emit expanded diagnostics (spread/bias statistics) during validation",
    )
    parser.add_argument(
        "--diagnostics-noise-threshold",
        type=float,
        help="Trigger a warning when the gradient noise ratio exceeds this value",
    )
    parser.add_argument(
        "--diagnostics-bias-threshold",
        type=float,
        help="Trigger a warning when validation bias magnitude exceeds this value",
    )
    parser.add_argument(
        "--diagnostics-error-threshold",
        type=float,
        help="Trigger a warning when validation absolute error exceeds this value",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        help="Warm start the supervised run from a pretraining checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_from_file(args.config)
    _apply_overrides(config, args)
    _ensure_wandb_defaults(config, config_path=args.config, disabled=args.no_wandb)
    run_training(config, pretrain_checkpoint_path=args.pretrain_checkpoint)


if __name__ == "__main__":
    main()

