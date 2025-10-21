"""Helpers that execute the MVP training→inference→monitoring loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..data.fixtures import FixtureConfig, build_fixture, write_fixture
from ..evaluation import OperationsSummary, OperationsThresholds, compile_operations_summary
from ..monitoring import DriftAlertThresholds, LiveMonitor, MonitoringSnapshot
from ..trading import AgentRunResult, MarketNNPlusUltraAgent
from ..training import (
    ExperimentConfig,
    TrainingRunResult,
    load_experiment_from_file,
    run_training,
)


DEFAULT_FIXTURE_CONFIG = FixtureConfig(
    symbols=["ALPHA", "BETA", "GAMMA"],
    rows=4_096,
    freq="30min",
    seed=11,
    start=datetime(2018, 1, 1),
    alt_features=2,
)


@dataclass(slots=True)
class MVPPipelineState:
    """Lightweight container for intermediate MVP run artefacts."""

    experiment_config: ExperimentConfig
    training: TrainingRunResult | None = None
    agent: AgentRunResult | None = None
    operations: OperationsSummary | None = None
    monitor: LiveMonitor | None = None
    monitoring_snapshot: MonitoringSnapshot | None = None


def ensure_fixture(
    output_path: Path,
    *,
    config: FixtureConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a SQLite fixture if it is missing and return the path."""

    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        return output_path

    fixture_config = config or DEFAULT_FIXTURE_CONFIG
    frames = build_fixture(fixture_config)
    write_fixture(frames, output_path)
    return output_path


def load_mvp_experiment(
    config_path: Path,
    *,
    max_epochs: int | None = None,
    limit_train_batches: float | None = None,
    limit_val_batches: float | None = None,
    batch_size: int | None = None,
    accelerator: str | None = None,
    devices: int | str | None = None,
) -> ExperimentConfig:
    """Load an experiment config and apply lightweight overrides for MVP runs."""

    experiment = load_experiment_from_file(config_path)

    trainer = experiment.trainer
    if max_epochs is not None:
        trainer.max_epochs = max_epochs
    if limit_train_batches is not None:
        trainer.limit_train_batches = limit_train_batches
    if limit_val_batches is not None:
        trainer.limit_val_batches = limit_val_batches
    if batch_size is not None:
        trainer.batch_size = batch_size
    if accelerator is not None:
        trainer.accelerator = accelerator
    if devices is not None:
        trainer.devices = devices

    return experiment


def run_mvp_training(
    experiment: ExperimentConfig,
    *,
    pretrain_checkpoint: Path | None = None,
) -> TrainingRunResult:
    """Execute supervised training using the provided experiment configuration."""

    return run_training(experiment, pretrain_checkpoint_path=pretrain_checkpoint)


def run_mvp_inference(
    experiment: ExperimentConfig,
    *,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
    evaluate: bool = True,
    return_column: str = "realised_return",
    benchmark_column: str | None = None,
) -> AgentRunResult:
    """Generate predictions (and optional metrics) for the MVP experiment."""

    agent = MarketNNPlusUltraAgent(
        experiment_config=experiment,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return agent.run(
        evaluate=evaluate,
        return_column=return_column,
        benchmark_column=benchmark_column,
    )


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format for '{path}'")


def summarise_operations(
    predictions: pd.DataFrame,
    *,
    thresholds: OperationsThresholds | None = None,
    trades_path: Path | pd.DataFrame | None = None,
    return_column: str = "realised_return",
    benchmark_column: str | None = None,
    capital_base: float = 1.0,
    tail_percentile: float = 5.0,
    trade_timestamp_col: str = "timestamp",
    trade_symbol_col: str = "symbol",
    trade_notional_col: str = "notional",
    trade_position_col: str = "position",
    trade_price_col: str = "price",
    trade_return_col: str = "pnl",
) -> OperationsSummary:
    """Compile operations-ready profitability and guardrail diagnostics."""

    trades_df: pd.DataFrame | None
    if trades_path is None:
        trades_df = None
    elif isinstance(trades_path, pd.DataFrame):
        trades_df = trades_path
    else:
        trades_df = _load_table(Path(trades_path))

    return compile_operations_summary(
        predictions,
        trades=trades_df,
        return_col=return_column,
        benchmark_col=benchmark_column,
        trade_timestamp_col=trade_timestamp_col,
        trade_symbol_col=trade_symbol_col,
        trade_notional_col=trade_notional_col,
        trade_position_col=trade_position_col,
        trade_price_col=trade_price_col,
        trade_return_col=trade_return_col,
        capital_base=capital_base,
        tail_percentile=tail_percentile,
        thresholds=thresholds,
    )


def extract_reference_returns(
    predictions: pd.DataFrame,
    *,
    return_column: str = "realised_return",
) -> np.ndarray:
    """Return a clean array of realised returns suitable for monitoring baselines."""

    if return_column not in predictions:
        raise ValueError(f"return column '{return_column}' missing from predictions")
    series = pd.to_numeric(predictions[return_column], errors="coerce").dropna()
    values = series.to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError("predictions do not contain any finite realised returns")
    return values


def build_monitor(
    reference_returns: Sequence[float] | np.ndarray,
    *,
    window_size: int = 512,
    drift_bins: int = 20,
    risk_thresholds: OperationsThresholds | None = None,
    drift_thresholds: DriftAlertThresholds | None = None,
) -> LiveMonitor:
    """Instantiate a live monitor that tracks risk and drift for new returns."""

    return LiveMonitor(
        reference_returns,
        window_size=window_size,
        drift_bins=drift_bins,
        risk_thresholds=risk_thresholds,
        drift_thresholds=drift_thresholds,
    )


def update_monitor(
    monitor: LiveMonitor,
    returns: Iterable[float],
) -> MonitoringSnapshot:
    """Feed returns into the live monitor and return the latest snapshot."""

    return monitor.update(returns)


__all__ = [
    "DEFAULT_FIXTURE_CONFIG",
    "MVPPipelineState",
    "ensure_fixture",
    "load_mvp_experiment",
    "run_mvp_training",
    "run_mvp_inference",
    "summarise_operations",
    "extract_reference_returns",
    "build_monitor",
    "update_monitor",
]
