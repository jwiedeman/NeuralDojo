"""CLI entrypoint for running masked time-series pretraining."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run masked time-series pretraining for Market NN Plus Ultra")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_from_file(args.config)
    if config.pretraining is None:
        raise SystemExit("Config must include a 'pretraining' section for self-supervised runs")
    _apply_overrides(config, args)
    result = run_pretraining(config)
    print(f"Best checkpoint stored at: {result['best_model_path']}")


if __name__ == "__main__":
    main()
