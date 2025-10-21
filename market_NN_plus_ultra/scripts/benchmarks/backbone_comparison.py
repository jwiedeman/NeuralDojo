"""Compare omni-scale and hybrid backbones across synthetic asset universes."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from market_nn_plus_ultra.data import FixtureConfig
from market_nn_plus_ultra.evaluation.backbone_comparison import (
    AssetUniverseSpec,
    generate_backbone_report,
    run_backbone_comparison,
)
from market_nn_plus_ultra.training import BenchmarkScenario, TrainerOverrides


def _default_universes() -> list[AssetUniverseSpec]:
    start = datetime(2024, 1, 1)
    return [
        AssetUniverseSpec(
            name="equities",
            label="Equities Basket",
            fixture=FixtureConfig(
                symbols=["EQ1", "EQ2", "EQ3"],
                rows=192,
                freq="15min",
                seed=101,
                start=start,
                alt_features=1,
            ),
        ),
        AssetUniverseSpec(
            name="crypto",
            label="Crypto Majors",
            fixture=FixtureConfig(
                symbols=["BTC", "ETH"],
                rows=224,
                freq="10min",
                seed=202,
                start=start,
                alt_features=0,
            ),
        ),
        AssetUniverseSpec(
            name="fx",
            label="FX Crosses",
            fixture=FixtureConfig(
                symbols=["EURUSD", "USDJPY", "GBPUSD"],
                rows=180,
                freq="30min",
                seed=303,
                start=start,
                alt_features=0,
            ),
        ),
    ]


def _default_scenarios(model_dim: int, depth: int, horizon: int) -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            architecture="hybrid_transformer",
            model_dim=model_dim,
            depth=depth,
            horizon=horizon,
            conv_dilations=(1, 2, 4),
            label="hybrid",
        ),
        BenchmarkScenario(
            architecture="omni_mixture",
            model_dim=model_dim,
            depth=depth,
            horizon=horizon,
            conv_dilations=(1, 2, 4),
            label="omni",
        ),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Directory for generated CSV and Markdown outputs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional override for the raw results CSV path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional override for the Markdown report path.",
    )
    parser.add_argument("--window-size", type=int, default=64, help="Sliding window size for training datasets.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon for the benchmarks.")
    parser.add_argument("--stride", type=int, default=4, help="Stride between training windows.")
    parser.add_argument("--model-dim", type=int, default=64, help="Base model dimension for both backbones.")
    parser.add_argument("--depth", type=int, default=2, help="Backbone depth for both architectures.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size used during training.")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2,
        help="Maximum number of epochs for each benchmark run.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=8.0,
        help="Limit on training batches to keep runs lightweight.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=8.0,
        help="Limit on validation batches for quick sweeps.",
    )
    parser.add_argument(
        "--universes",
        type=str,
        help="Comma separated list of default universe names to run (default: all).",
    )
    parser.add_argument(
        "--regenerate-fixtures",
        action="store_true",
        help="Regenerate fixture databases even if cached copies exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    universes = _default_universes()
    if args.universes:
        allowed = {name.strip().lower() for name in args.universes.split(",") if name.strip()}
        universes = [u for u in universes if u.name.lower() in allowed]
        if not universes:
            raise SystemExit("No matching universes selected; nothing to benchmark.")

    scenarios = _default_scenarios(args.model_dim, args.depth, args.horizon)
    overrides = TrainerOverrides(
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        batch_size=args.batch_size,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=5,
    )

    fixture_dir = output_dir / "fixtures"
    frame = run_backbone_comparison(
        universes,
        scenarios=scenarios,
        trainer_overrides=overrides,
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        base_model_dim=args.model_dim,
        base_depth=args.depth,
        base_batch_size=args.batch_size,
        regenerate_fixtures=args.regenerate_fixtures,
        fixture_root=fixture_dir,
    )

    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)

    csv_path = args.csv or (output_dir / "backbone_comparison.csv")
    frame.to_csv(csv_path, index=False)

    report = generate_backbone_report(frame)
    report_path = args.report or (output_dir / "backbone_comparison.md")
    report_path.write_text(report)

    print(f"Wrote raw benchmark results to {csv_path}")
    print(f"Wrote Markdown summary to {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
