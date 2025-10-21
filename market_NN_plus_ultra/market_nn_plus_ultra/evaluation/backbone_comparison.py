from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
import tempfile

import pandas as pd

from ..data import FixtureConfig, build_fixture, write_fixture
from ..data.sqlite_loader import SQLiteMarketDataset, SQLiteMarketSource
from ..training import (
    BenchmarkScenario,
    TrainerOverrides,
    flatten_benchmark_result,
    prepare_config_for_scenario,
    run_training,
)
from ..training.config import (
    CalibrationConfig,
    DataConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    MarketStateConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from .benchmarking import (
    architecture_leaderboard,
    dataframe_to_markdown,
    summarise_architecture_performance,
    summaries_to_frame,
)


@dataclass(slots=True)
class AssetUniverseSpec:
    """Synthetic asset universe used for benchmarking."""

    name: str
    fixture: FixtureConfig
    label: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.name


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_base_config(
    *,
    sqlite_path: Path,
    feature_dim: int,
    horizon: int,
    window_size: int,
    stride: int,
    batch_size: int,
    checkpoint_dir: Path,
    seed: int,
    model_dim: int,
    depth: int,
    conv_dilations: tuple[int, ...],
    symbols: Sequence[str],
) -> ExperimentConfig:
    data_config = DataConfig(
        sqlite_path=sqlite_path,
        symbol_universe=list(symbols),
        indicators={},
        alternative_data=[],
        resample_rule=None,
        tz_convert=None,
        feature_set=None,
        target_columns=["close"],
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        normalise=True,
        val_fraction=0.25,
    )

    model_config = ModelConfig(
        feature_dim=feature_dim,
        model_dim=model_dim,
        depth=depth,
        heads=2,
        dropout=0.1,
        conv_kernel_size=3,
        conv_dilations=conv_dilations,
        horizon=horizon,
        output_dim=1,
        architecture="hybrid_transformer",
        ff_mult=2,
        num_experts=2,
        router_dropout=0.0,
        ssm_state_dim=64,
        ssm_kernel_size=7,
        coarse_factor=2,
        cross_every=2,
        max_seq_len=4096,
        use_rotary_embeddings=True,
        rope_theta=10000.0,
        gradient_checkpointing=False,
        calibration=CalibrationConfig(enabled=False),
        market_state=MarketStateConfig(enabled=False),
    )

    optimizer_config = OptimizerConfig(lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.95))

    trainer_config = TrainerConfig(
        batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
        max_epochs=2,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        matmul_precision=None,
        log_every_n_steps=5,
        checkpoint_dir=checkpoint_dir,
        monitor_metric="val/loss",
        monitor_mode="min",
        save_top_k=1,
        num_sanity_val_steps=0,
        limit_train_batches=8,
        limit_val_batches=8,
    )

    diagnostics_config = DiagnosticsConfig(enabled=False, log_interval=5, profile=False)

    return ExperimentConfig(
        seed=seed,
        data=data_config,
        model=model_config,
        optimizer=optimizer_config,
        trainer=trainer_config,
        diagnostics=diagnostics_config,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_tags=tuple(),
        wandb_offline=True,
    )


def _load_feature_dimension(sqlite_path: Path) -> int:
    dataset = SQLiteMarketDataset(SQLiteMarketSource(str(sqlite_path)), validate=False)
    frame = dataset.load()
    if frame.empty:
        raise ValueError(f"Fixture at {sqlite_path} produced no rows")
    return len(frame.columns)


def _execute_comparison(
    universes: Sequence[AssetUniverseSpec],
    *,
    fixture_dir: Path,
    regenerate: bool,
    scenarios: Sequence[BenchmarkScenario],
    trainer_overrides: TrainerOverrides,
    window_size: int,
    horizon: int,
    stride: int,
    base_model_dim: int,
    base_depth: int,
    conv_dilations: tuple[int, ...],
    batch_size: int,
) -> pd.DataFrame:
    _ensure_directory(fixture_dir)
    checkpoint_root = fixture_dir / "checkpoints"
    _ensure_directory(checkpoint_root)

    rows: list[dict[str, object]] = []
    for universe in universes:
        sqlite_path = fixture_dir / f"{universe.name}.db"
        if regenerate or not sqlite_path.exists():
            frames = build_fixture(universe.fixture)
            write_fixture(frames, sqlite_path)

        feature_dim = _load_feature_dimension(sqlite_path)
        base_config = _build_base_config(
            sqlite_path=sqlite_path,
            feature_dim=feature_dim,
            horizon=horizon,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_root / universe.name,
            seed=universe.fixture.seed,
            model_dim=base_model_dim,
            depth=base_depth,
            conv_dilations=conv_dilations,
            symbols=universe.fixture.symbols,
        )

        for scenario in scenarios:
            scenario_label = scenario.label or scenario.architecture
            labelled = replace(
                scenario,
                label=f"{universe.name}-{scenario_label}",
            )
            config = prepare_config_for_scenario(
                base_config,
                labelled,
                overrides=trainer_overrides,
                disable_wandb=True,
            )
            start = time.perf_counter()
            result = run_training(config)
            duration = time.perf_counter() - start
            row = flatten_benchmark_result(labelled, result, duration_seconds=duration)
            row["universe"] = universe.display_name
            row["sqlite_path"] = str(sqlite_path)
            rows.append(row)

    frame = pd.DataFrame(rows)
    return frame


def run_backbone_comparison(
    universes: Sequence[AssetUniverseSpec],
    *,
    scenarios: Sequence[BenchmarkScenario] | None = None,
    trainer_overrides: TrainerOverrides | None = None,
    window_size: int = 64,
    horizon: int = 3,
    stride: int = 4,
    base_model_dim: int = 64,
    base_depth: int = 2,
    conv_dilations: tuple[int, ...] = (1, 2, 4),
    omni_conv_dilations: tuple[int, ...] | None = None,
    base_batch_size: int = 16,
    regenerate_fixtures: bool = False,
    fixture_root: Path | None = None,
) -> pd.DataFrame:
    """Run supervised training benchmarks for multiple asset universes."""

    if not universes:
        raise ValueError("At least one asset universe must be provided")

    overrides = trainer_overrides or TrainerOverrides(
        max_epochs=2,
        limit_train_batches=8,
        limit_val_batches=8,
        batch_size=base_batch_size,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=5,
    )

    scenario_list: list[BenchmarkScenario]
    if scenarios is None:
        omni_dilations = omni_conv_dilations or conv_dilations
        scenario_list = [
            BenchmarkScenario(
                architecture="hybrid_transformer",
                model_dim=base_model_dim,
                depth=base_depth,
                horizon=horizon,
                conv_dilations=tuple(conv_dilations),
                label="hybrid",
            ),
            BenchmarkScenario(
                architecture="omni_mixture",
                model_dim=base_model_dim,
                depth=base_depth,
                horizon=horizon,
                conv_dilations=tuple(omni_dilations),
                label="omni",
            ),
        ]
    else:
        scenario_list = list(scenarios)

    if fixture_root is not None:
        _ensure_directory(fixture_root)
        frame = _execute_comparison(
            universes,
            fixture_dir=fixture_root,
            regenerate=regenerate_fixtures,
            scenarios=scenario_list,
            trainer_overrides=overrides,
            window_size=window_size,
            horizon=horizon,
            stride=stride,
            base_model_dim=base_model_dim,
            base_depth=base_depth,
            conv_dilations=tuple(conv_dilations),
            batch_size=base_batch_size,
        )
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame = _execute_comparison(
                universes,
                fixture_dir=Path(tmp_dir),
                regenerate=True,
                scenarios=scenario_list,
                trainer_overrides=overrides,
                window_size=window_size,
                horizon=horizon,
                stride=stride,
                base_model_dim=base_model_dim,
                base_depth=base_depth,
                conv_dilations=tuple(conv_dilations),
                batch_size=base_batch_size,
            )

    if frame.empty:
        raise RuntimeError("Backbone comparison produced no results")

    frame["universe"] = frame["universe"].astype(str)
    frame.sort_values(["universe", "architecture"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def generate_backbone_report(
    frame: pd.DataFrame,
    *,
    metric: str = "metric_val_loss",
    profitability_metric: str = "profitability_roi",
) -> str:
    """Return a Markdown report summarising benchmark outcomes."""

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    summaries = summarise_architecture_performance(
        frame,
        metric=metric,
        higher_is_better=False,
        profitability_metric=profitability_metric,
    )
    summary_frame = summaries_to_frame(summaries)
    leaderboard = architecture_leaderboard(
        frame,
        metric=metric,
        higher_is_better=False,
        group_by=["universe"],
        top_k=1,
    )

    report_lines = [
        "# Omni vs. Hybrid Backbone Benchmark",
        "",
        f"Generated: {timestamp}",
        "",
        "## Architecture Summary",
        dataframe_to_markdown(summary_frame),
        "",
        "## Per-Universe Leaders (val/loss)",
        dataframe_to_markdown(leaderboard),
    ]

    if profitability_metric in frame.columns:
        try:
            profitability_leaderboard = architecture_leaderboard(
                frame,
                metric=profitability_metric,
                higher_is_better=True,
                group_by=["universe"],
                top_k=1,
            )
        except (KeyError, ValueError):
            profitability_leaderboard = None
        if profitability_leaderboard is not None and not profitability_leaderboard.empty:
            report_lines.extend(
                [
                    "",
                    f"## Per-Universe Leaders ({profitability_metric})",
                    dataframe_to_markdown(profitability_leaderboard),
                ]
            )

    return "\n".join(report_lines)


__all__ = [
    "AssetUniverseSpec",
    "generate_backbone_report",
    "run_backbone_comparison",
]
