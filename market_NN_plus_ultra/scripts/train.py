"""CLI entrypoint for training Market NN Plus Ultra models."""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_from_file(args.config)
    _apply_overrides(config, args)
    run_training(config)


if __name__ == "__main__":
    main()

