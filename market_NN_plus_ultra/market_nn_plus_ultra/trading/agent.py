"""Inference and signal generation utilities for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from ..data import FeatureRegistry, SQLiteMarketDataset, SlidingWindowDataset
from ..data.sqlite_loader import SQLiteMarketSource
from ..evaluation.metrics import risk_metrics
from ..utils.reporting import sanitize_metrics
from ..training import ExperimentConfig, MarketLightningModule


@dataclass(slots=True)
class AgentRunResult:
    """Container for agent predictions and optional evaluation metrics."""

    predictions: pd.DataFrame
    metrics: Optional[dict[str, float]] = None


class MarketNNPlusUltraAgent:
    """High-level helper that mirrors the classic market agent workflow."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        checkpoint_path: Optional[Path] = None,
        device: str | torch.device = "cpu",
        batch_size: Optional[int] = None,
    ) -> None:
        self.config = experiment_config
        self.device = torch.device(device)
        self.batch_size = batch_size or experiment_config.trainer.batch_size
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is not None:
            module = MarketLightningModule.from_checkpoint(
                checkpoint_path=checkpoint_path,
                model_config=experiment_config.model,
                optimizer_config=experiment_config.optimizer,
                map_location=self.device,
            )
        else:
            module = MarketLightningModule(experiment_config.model, experiment_config.optimizer)
        self.model = module.to(self.device)
        self.model.eval()

        self.registry = FeatureRegistry()
        self._dataset: Optional[SlidingWindowDataset] = None
        self._feature_columns: list[str] = []

    def prepare_dataset(self) -> SlidingWindowDataset:
        """Load SQLite data, enrich features, and create a sliding-window dataset."""

        data_cfg = self.config.data
        source = SQLiteMarketSource(path=str(data_cfg.sqlite_path))
        dataset = SQLiteMarketDataset(
            source=source,
            symbol_universe=data_cfg.symbol_universe,
            indicators=data_cfg.indicators,
            resample_rule=data_cfg.resample_rule,
            tz_convert=data_cfg.tz_convert,
        )
        panel = dataset.as_panel()
        pipeline = self.registry.build_pipeline(data_cfg.feature_set)
        enriched = pipeline.transform_panel(panel)
        if data_cfg.feature_set:
            feature_columns = [f for f in data_cfg.feature_set if f in enriched.columns]
        else:
            feature_columns = [c for c in enriched.columns if c not in ("symbol",)]
        sliding = SlidingWindowDataset(
            panel=enriched,
            feature_columns=feature_columns,
            target_columns=data_cfg.target_columns,
            window_size=data_cfg.window_size,
            horizon=data_cfg.horizon,
            stride=data_cfg.stride,
            normalise=data_cfg.normalise,
        )
        if len(feature_columns) != self.config.model.feature_dim:
            raise ValueError(
                "Configured feature_dim does not match engineered features. "
                f"Expected {self.config.model.feature_dim}, got {len(feature_columns)}"
            )
        self._dataset = sliding
        self._feature_columns = feature_columns
        return sliding

    def generate_signals(self, dataset: Optional[SlidingWindowDataset] = None) -> pd.DataFrame:
        """Run the model over the dataset and return a tidy prediction frame."""

        dataset = dataset or self._dataset or self.prepare_dataset()
        rows: list[dict[str, object]] = []
        horizon = self.config.model.horizon
        output_dim = self.config.model.output_dim

        for start in range(0, len(dataset), self.batch_size or 1):
            batch_indices = list(range(start, min(start + (self.batch_size or 1), len(dataset))))
            features_batch = []
            targets_batch = []
            metadata_batch = []
            for idx in batch_indices:
                sample = dataset[idx]
                metadata = dataset.get_metadata(idx)
                features_batch.append(sample["features"])
                targets_batch.append(sample["targets"])
                metadata_batch.append(metadata)

            features_tensor = torch.stack(features_batch, dim=0).to(self.device)
            with torch.no_grad():
                preds_tensor = self.model(features_tensor).detach().cpu()
            targets_tensor = torch.stack(targets_batch, dim=0).detach().cpu()

            preds_np = preds_tensor.numpy()
            targets_np = targets_tensor.numpy()

            for batch_pos, metadata in enumerate(metadata_batch):
                preds = preds_np[batch_pos]
                targets = targets_np[batch_pos]

                if preds.ndim == 1:
                    preds = preds.reshape(horizon, -1)
                if preds.shape[0] != horizon:
                    raise ValueError(
                        f"Model produced horizon {preds.shape[0]} but config expects {horizon}"
                    )

                row: dict[str, object] = {
                    "symbol": metadata.symbol,
                    "window_end": metadata.input_timestamps[-1],
                    "start_index": metadata.start_index,
                }
                for step, timestamp in enumerate(metadata.target_timestamps):
                    row[f"target_timestamp_{step}"] = timestamp

                if preds.shape[1] == 1:
                    for step in range(horizon):
                        row[f"pred_step_{step}"] = float(preds[step, 0])
                else:
                    for step in range(horizon):
                        for dim in range(preds.shape[1]):
                            row[f"pred_step_{step}_dim_{dim}"] = float(preds[step, dim])

                if targets.ndim == 1:
                    targets = targets.reshape(horizon, -1)
                for col_idx, col_name in enumerate(dataset.target_columns):
                    for step in range(min(horizon, targets.shape[0])):
                        row[f"actual_{col_name}_step_{step}"] = float(targets[step, col_idx])

                if output_dim == 1:
                    row["predicted_return"] = float(preds[:, 0].mean())
                else:
                    row["predicted_return"] = float(preds.mean())

                if targets.size:
                    row["realised_return"] = float(targets[:, 0].mean())

                rows.append(row)

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(["symbol", "window_end"]).reset_index(drop=True)
        return frame

    def evaluate_predictions(self, predictions: pd.DataFrame, return_column: str = "realised_return") -> dict[str, float]:
        """Compute risk metrics on a column of realised returns."""

        if return_column not in predictions:
            raise ValueError(f"Column '{return_column}' not found in predictions DataFrame")
        returns = predictions[return_column].to_numpy()
        metrics = risk_metrics(returns)
        return sanitize_metrics(metrics)

    def run(self, evaluate: bool = True, return_column: str = "realised_return") -> AgentRunResult:
        """Convenience wrapper that prepares data, generates signals, and evaluates."""

        dataset = self.prepare_dataset()
        predictions = self.generate_signals(dataset)
        metrics = None
        if evaluate and not predictions.empty and return_column in predictions.columns:
            metrics = self.evaluate_predictions(predictions, return_column=return_column)
        return AgentRunResult(predictions=predictions, metrics=metrics)

    @property
    def feature_columns(self) -> list[str]:
        """Return the feature column names used during inference."""

        return list(self._feature_columns)

    @property
    def dataset(self) -> Optional[SlidingWindowDataset]:
        """Expose the cached dataset when it has been prepared."""

        return self._dataset


__all__ = ["MarketNNPlusUltraAgent", "AgentRunResult"]
