"""CLI for PPO fine-tuning of the Market NN Plus Ultra agent."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from market_nn_plus_ultra.cli.reinforcement import (
    apply_reinforcement_overrides,
    register_reinforcement_arguments,
)
from market_nn_plus_ultra.training import (
    ReinforcementConfig,
    load_experiment_from_file,
    run_reinforcement_finetuning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO fine-tuning for the Plus Ultra agent")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--checkpoint",
        "--warm-start-checkpoint",
        dest="checkpoint",
        type=Path,
        help="Optional supervised checkpoint to warm-start from",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        help="Self-supervised checkpoint to initialise the backbone before PPO",
    )
    parser.add_argument(
        "--warm-start-tuning",
        action="store_true",
        help="Require that a warm-start checkpoint is supplied before launching PPO",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier (cpu, cuda:0, etc.)")
    parser.add_argument("--updates", type=int, help="Override the number of PPO updates to perform")
    parser.add_argument("--steps-per-rollout", type=int, help="Override rollout size in samples")
    parser.add_argument("--policy-epochs", type=int, help="Override the number of policy epochs per update")
    parser.add_argument("--minibatch-size", type=int, help="Override PPO minibatch size")
    parser.add_argument("--gamma", type=float, help="Discount factor for GAE")
    parser.add_argument("--gae-lambda", type=float, help="Lambda parameter for GAE")
    parser.add_argument("--clip-ratio", type=float, help="Clipping ratio for PPO surrogate objective")
    parser.add_argument("--value-coef", type=float, help="Coefficient for value loss")
    parser.add_argument("--entropy-coef", type=float, help="Coefficient for entropy bonus")
    parser.add_argument("--learning-rate", type=float, help="Override PPO learning rate")
    parser.add_argument("--max-grad-norm", type=float, help="Gradient clipping norm")
    parser.add_argument("--rollout-workers", type=int, help="Number of parallel rollout workers")
    parser.add_argument("--worker-device", type=str, help="Device identifier for rollout workers")
    register_reinforcement_arguments(parser)
    return parser.parse_args()


def apply_overrides(config: ReinforcementConfig, args: argparse.Namespace) -> ReinforcementConfig:
    updated = replace(config)
    if args.updates is not None:
        updated.total_updates = args.updates
    if args.steps_per_rollout is not None:
        updated.steps_per_rollout = args.steps_per_rollout
    if args.policy_epochs is not None:
        updated.policy_epochs = args.policy_epochs
    if args.minibatch_size is not None:
        updated.minibatch_size = args.minibatch_size
    if args.gamma is not None:
        updated.gamma = args.gamma
    if args.gae_lambda is not None:
        updated.gae_lambda = args.gae_lambda
    if args.clip_ratio is not None:
        updated.clip_ratio = args.clip_ratio
    if args.value_coef is not None:
        updated.value_coef = args.value_coef
    if args.entropy_coef is not None:
        updated.entropy_coef = args.entropy_coef
    if args.learning_rate is not None:
        updated.learning_rate = args.learning_rate
    if args.max_grad_norm is not None:
        updated.max_grad_norm = args.max_grad_norm
    if args.rollout_workers is not None:
        updated.rollout_workers = args.rollout_workers
    if args.worker_device is not None:
        updated.worker_device = args.worker_device
    return apply_reinforcement_overrides(updated, args)


def main() -> None:
    args = parse_args()
    experiment = load_experiment_from_file(args.config)
    reinforcement = experiment.reinforcement or ReinforcementConfig()
    reinforcement = apply_overrides(reinforcement, args)
    if args.warm_start_tuning and not (args.checkpoint or args.pretrain_checkpoint):
        raise SystemExit("--warm-start-tuning requires a warm-start checkpoint to be provided")

    if args.checkpoint and args.pretrain_checkpoint:
        raise SystemExit("Specify either --checkpoint or --pretrain-checkpoint, not both")

    result = run_reinforcement_finetuning(
        experiment,
        reinforcement_config=reinforcement,
        checkpoint_path=args.checkpoint,
        pretrain_checkpoint_path=args.pretrain_checkpoint,
        device=args.device,
    )
    print("\n=== PPO Fine-tuning Summary ===")
    for update in result.updates:
        print(
            f"Update {update.update:03d} | Reward: {update.mean_reward:.6f} | "
            f"Std: {update.reward_std:.6f} | Policy Loss: {update.policy_loss:.6f} | "
            f"Value Loss: {update.value_loss:.6f} | Entropy: {update.entropy:.6f} | "
            f"Samples: {update.samples} | Throughput: {update.samples_per_second:.2f} seq/s"
        )

    if result.evaluation_metrics:
        print("\n--- Deterministic Evaluation Metrics ---")
        for name in sorted(result.evaluation_metrics):
            value = result.evaluation_metrics[name]
            print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()

