import argparse
import json
from pathlib import Path

from market_nn_plus_ultra.training import load_experiment_from_file
from market_nn_plus_ultra.utils import profile_backbone_throughput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Plus Ultra backbone throughput")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Experiment YAML path")
    parser.add_argument("--architecture", type=str, help="Override the model architecture")
    parser.add_argument("--model-dim", type=int, help="Override the model hidden dimension")
    parser.add_argument("--feature-dim", type=int, help="Override the expected feature dimension")
    parser.add_argument("--depth", type=int, help="Override the number of backbone layers")
    parser.add_argument("--heads", type=int, help="Override attention head count")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used during profiling")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length used during profiling")
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier (cpu, cuda:0, mps)")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Number of warm-up passes before timing")
    parser.add_argument("--measure-steps", type=int, default=5, help="Number of timed passes to average")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_from_file(args.config)
    model = experiment.model

    if args.architecture is not None:
        model.architecture = args.architecture.lower()
    if args.model_dim is not None:
        model.model_dim = args.model_dim
    if args.feature_dim is not None:
        model.feature_dim = args.feature_dim
    if args.depth is not None:
        model.depth = args.depth
    if args.heads is not None:
        model.heads = args.heads

    report = profile_backbone_throughput(
        model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )

    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
