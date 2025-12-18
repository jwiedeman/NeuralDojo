"""Lightning-powered training loop for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from ..data import FeatureRegistry, SQLiteMarketDataset, SlidingWindowDataset
from ..data.alternative_data import AlternativeDataSpec
from ..data.sqlite_loader import SQLiteMarketSource
from ..models.temporal_transformer import TemporalBackbone, TemporalBackboneConfig, TemporalPolicyHead
from ..models.temporal_fusion import TemporalFusionConfig, TemporalFusionTransformer
from ..models.omni_mixture import MarketOmniBackbone, OmniBackboneConfig
from ..models.multi_scale import MultiScaleBackbone, MultiScaleBackboneConfig
from ..models.moe_transformer import MixtureOfExpertsBackbone, MixtureOfExpertsConfig
from ..models.state_space import StateSpaceBackbone, StateSpaceConfig
from ..models.losses import CompositeTradingLoss
from ..models.calibration import CalibratedPolicyHead, CalibrationHeadOutput
from ..models.market_state import MarketStateEmbedding, MarketStateMetadata
from ..trading.pnl import TradingCosts, differentiable_pnl, price_to_returns
from ..trading.risk import compute_risk_metrics
from ..utils.wandb import maybe_create_wandb_logger
from ..utils.reporting import write_metrics_report
from .config import (
    CalibrationConfig,
    CurriculumConfig,
    CurriculumStage,
    DataConfig,
    GuardrailConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainingConfig,
    ReplayBufferConfig,
    ReinforcementConfig,
    TrainerConfig,
    MarketStateConfig,
)
from .checkpoints import load_backbone_from_checkpoint
from .curriculum import (
    CurriculumCallback,
    CurriculumParameters,
    CurriculumScheduler,
)
from .diagnostics import TrainingDiagnosticsCallback


logger = logging.getLogger(__name__)
_IS_WINDOWS = sys.platform.startswith("win")


@dataclass(slots=True)
class TrainingRunResult:
    """Structured summary returned after running supervised training."""

    best_model_path: str
    logged_metrics: dict[str, float]
    dataset_summary: dict[str, int]
    profitability_summary: dict[str, float] = field(default_factory=dict)
    profitability_reports: dict[str, str] = field(default_factory=dict)


def _parameter_counts(module: pl.LightningModule) -> tuple[int, int]:
    """Return trainable and total parameter counts for logging."""

    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


class MarketLightningModule(pl.LightningModule):
    """Lightning module combining the backbone, policy head, and state embeddings."""

    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        *,
        market_state_metadata: MarketStateMetadata | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.save_hyperparameters({"model": asdict(model_config), "optimizer": asdict(optimizer_config)})
        self.market_state_embedding: MarketStateEmbedding | None = None
        self._state_indices: tuple[int, ...] | None = None
        self._market_state_metadata: MarketStateMetadata | None = None
        architecture = model_config.architecture.lower()
        if architecture in {"hybrid_transformer", "temporal_transformer"}:
            backbone_config = TemporalBackboneConfig(
                feature_dim=model_config.feature_dim,
                model_dim=model_config.model_dim,
                depth=model_config.depth,
                heads=model_config.heads,
                dropout=model_config.dropout,
                conv_kernel_size=model_config.conv_kernel_size,
                conv_dilations=model_config.conv_dilations,
                max_seq_len=model_config.max_seq_len,
                use_rotary_embeddings=model_config.use_rotary_embeddings,
                rope_theta=model_config.rope_theta,
            )
            self.backbone = TemporalBackbone(backbone_config)
        elif architecture in {"temporal_fusion", "tft"}:
            fusion_config = TemporalFusionConfig(
                feature_dim=model_config.feature_dim,
                hidden_dim=model_config.model_dim,
                num_heads=model_config.heads,
                dropout=model_config.dropout,
                num_encoder_layers=model_config.encoder_layers or model_config.depth,
                num_decoder_layers=model_config.decoder_layers or model_config.depth,
                horizon=model_config.horizon,
                max_seq_len=model_config.max_seq_len,
            )
            self.backbone = TemporalFusionTransformer(fusion_config)
        elif architecture in {"moe", "moe_transformer", "mixture_of_experts"}:
            moe_config = MixtureOfExpertsConfig(
                feature_dim=model_config.feature_dim,
                model_dim=model_config.model_dim,
                depth=model_config.depth,
                heads=model_config.heads,
                dropout=model_config.dropout,
                num_experts=model_config.num_experts,
                ff_mult=model_config.ff_mult,
                router_dropout=model_config.router_dropout,
                conv_kernel_size=model_config.conv_kernel_size,
                conv_dilations=model_config.conv_dilations,
                max_seq_len=model_config.max_seq_len,
            )
            self.backbone = MixtureOfExpertsBackbone(moe_config)
        elif architecture in {"omni", "omni_mixture", "omni_backbone"}:
            backbone_config = OmniBackboneConfig(
                feature_dim=model_config.feature_dim,
                model_dim=model_config.model_dim,
                depth=model_config.depth,
                heads=model_config.heads,
                dropout=model_config.dropout,
                ff_mult=model_config.ff_mult,
                ssm_state_dim=model_config.ssm_state_dim,
                ssm_kernel_size=model_config.ssm_kernel_size,
                conv_kernel_size=model_config.conv_kernel_size,
                conv_dilations=model_config.conv_dilations,
                coarse_factor=model_config.coarse_factor,
                cross_every=model_config.cross_every,
                max_seq_len=model_config.max_seq_len,
                gradient_checkpointing=model_config.gradient_checkpointing,
            )
            self.backbone = MarketOmniBackbone(backbone_config)
        elif architecture in {"multi_scale", "multiscale", "hierarchical"}:
            backbone_config = MultiScaleBackboneConfig(
                feature_dim=model_config.feature_dim,
                model_dim=model_config.model_dim,
                scales=tuple(model_config.scale_factors),
                depth_per_scale=model_config.scale_depth or model_config.depth,
                heads=model_config.heads,
                dropout=model_config.dropout,
                conv_kernel_size=model_config.conv_kernel_size,
                conv_dilations=model_config.conv_dilations,
                max_seq_len=model_config.max_seq_len,
                fusion_heads=model_config.fusion_heads,
                use_rotary_embeddings=model_config.use_rotary_embeddings,
                rope_theta=model_config.rope_theta,
            )
            self.backbone = MultiScaleBackbone(backbone_config)
        elif architecture in {"state_space", "ssm", "s4"}:
            ssm_config = StateSpaceConfig(
                feature_dim=model_config.feature_dim,
                model_dim=model_config.model_dim,
                depth=model_config.depth,
                state_dim=model_config.ssm_state_dim,
                kernel_size=model_config.ssm_kernel_size,
                dropout=model_config.dropout,
                ff_mult=model_config.ff_mult,
                max_seq_len=model_config.max_seq_len,
            )
            self.backbone = StateSpaceBackbone(ssm_config)
        else:
            raise ValueError(f"Unknown architecture '{model_config.architecture}'")
        self.head = self._build_head(model_config.horizon)
        self.loss_fn = CompositeTradingLoss()
        self._latest_head_output: CalibrationHeadOutput | None = None
        self._profitability_buffer: dict[str, float] | None = None
        self._latest_profitability: dict[str, float] | None = None

        if self.model_config.market_state.enabled:
            metadata = market_state_metadata
            if metadata is None or metadata.feature_count() == 0:
                logger.warning(
                    "Market-state embeddings enabled but no metadata provided; continuing without embeddings."
                )
            else:
                selected = metadata.select(self.model_config.market_state.include)
                if selected.feature_count() == 0:
                    logger.warning(
                        "Market-state embeddings enabled but none of the requested features %s are available.",
                        self.model_config.market_state.include,
                    )
                else:
                    self.market_state_embedding = MarketStateEmbedding(
                        selected,
                        embedding_dim=self.model_config.market_state.embedding_dim,
                        dropout=self.model_config.market_state.dropout,
                    )
                    self._state_indices = selected.indices()
                    self._market_state_metadata = selected

    def _compute_state_embedding(
        self,
        features: torch.Tensor,
        state_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.market_state_embedding is not None
        if state_tokens is None:
            return self.market_state_embedding.zero_state(features)
        tokens = state_tokens
        if tokens.dtype != torch.long:
            tokens = tokens.to(dtype=torch.long)
        if self._state_indices:
            index = torch.as_tensor(self._state_indices, device=tokens.device, dtype=torch.long)
            tokens = torch.index_select(tokens, dim=-1, index=index)
        tokens = tokens.to(device=features.device)
        embeddings = self.market_state_embedding(tokens)
        return embeddings.to(features.dtype)

    def forward(
        self,
        features: torch.Tensor,
        *,
        state_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_feat_dim = features.shape[-1]
        if self.market_state_embedding is not None:
            state_embed = self._compute_state_embedding(features, state_tokens)
            features = torch.cat([features, state_embed], dim=-1)
            if not hasattr(self, '_logged_dims'):
                logger.info(
                    "Forward pass dims: base_features=%d, state_embed=%d, total=%d, model_expects=%d",
                    base_feat_dim, state_embed.shape[-1], features.shape[-1], self.model_config.feature_dim
                )
                self._logged_dims = True
        hidden = self.backbone(features)
        head_output = self.head(hidden)
        if isinstance(head_output, CalibrationHeadOutput):
            self._latest_head_output = head_output
            return head_output.prediction
        self._latest_head_output = None
        return head_output

    def update_horizon(self, horizon: int) -> None:
        """Refresh the policy head when the training horizon changes."""

        if horizon == self.model_config.horizon:
            return
        self.model_config.horizon = horizon
        self.hparams["model"]["horizon"] = horizon
        new_head = self._build_head(horizon)
        self.head = new_head.to(self.device)
        self._latest_head_output = None

    def _build_head(self, horizon: int) -> nn.Module:
        if self.model_config.calibration.enabled:
            return CalibratedPolicyHead(
                self.model_config.model_dim,
                horizon,
                self.model_config.output_dim,
                quantile_levels=self.model_config.calibration.quantiles,
                dirichlet_temperature=self.model_config.calibration.temperature,
                min_concentration=self.model_config.calibration.min_concentration,
            )
        return TemporalPolicyHead(
            self.model_config.model_dim,
            horizon,
            self.model_config.output_dim,
        )

    @property
    def latest_head_output(self) -> CalibrationHeadOutput | None:
        """Return the most recent head output, if calibration is enabled."""

        return self._latest_head_output

    @property
    def latest_profitability_summary(self) -> dict[str, float] | None:
        """Return the most recently aggregated profitability metrics."""

        return None if self._latest_profitability is None else dict(self._latest_profitability)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        map_location: str | torch.device | None = None,
    ) -> "MarketLightningModule":
        """Instantiate a module and restore weights from a checkpoint."""

        module = cls(model_config, optimizer_config)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("state_dict", checkpoint)
        module.load_state_dict(state_dict)
        module.eval()
        return module

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds = self(batch["features"], state_tokens=batch.get("state_tokens"))
        targets = batch["targets"]
        reference = batch.get("reference")
        loss = self.loss_fn(preds, targets, reference=reference)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self(batch["features"], state_tokens=batch.get("state_tokens"))
        reference = batch.get("reference")
        loss = self.loss_fn(preds, batch["targets"], reference=reference)
        self.log("val/loss", loss, prog_bar=True)
        pnl_series = self._compute_pnl_series(preds, batch["targets"], reference)
        if pnl_series is not None:
            self._accumulate_profitability(pnl_series)

    def _compute_pnl_series(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        reference: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return per-sample PnL series used for profitability diagnostics."""

        if targets.ndim != 3 or preds.ndim != 3:
            logger.warning(
                "Skipping profitability accumulation; expected 3D tensors got preds=%s targets=%s",
                tuple(preds.shape),
                tuple(targets.shape),
            )
            return None

        try:
            future_returns = price_to_returns(targets.detach(), reference.detach() if isinstance(reference, torch.Tensor) else None)
        except ValueError as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to compute future returns for profitability summary: %s", exc)
            return None

        pnl_series = differentiable_pnl(
            preds.detach(),
            future_returns.detach(),
            costs=self.loss_fn.trading_costs,
            activation=self.loss_fn.activation,
        )
        return pnl_series.detach()

    def _accumulate_profitability(self, pnl_series: torch.Tensor) -> None:
        if self._profitability_buffer is None or pnl_series.ndim != 2:
            return

        pnl = pnl_series.detach()
        metrics = compute_risk_metrics(pnl, dim=-1)
        roi = pnl.sum(dim=-1)

        self._profitability_buffer["roi_sum"] += float(roi.sum().item())
        self._profitability_buffer["sharpe_sum"] += float(metrics.sharpe.sum().item())
        self._profitability_buffer["drawdown_sum"] += float(metrics.drawdown.sum().item())
        self._profitability_buffer["sample_count"] += float(pnl.size(0))

    def on_validation_epoch_start(self) -> None:
        self._profitability_buffer = {
            "roi_sum": 0.0,
            "sharpe_sum": 0.0,
            "drawdown_sum": 0.0,
            "sample_count": 0.0,
        }

    def on_validation_epoch_end(self) -> None:
        if not self._profitability_buffer:
            return
        count = self._profitability_buffer["sample_count"]
        if count <= 0:
            self._latest_profitability = None
            return
        summary = {
            "roi": self._profitability_buffer["roi_sum"] / count,
            "sharpe": self._profitability_buffer["sharpe_sum"] / count,
            "max_drawdown": self._profitability_buffer["drawdown_sum"] / count,
        }
        self._latest_profitability = summary
        for name, value in summary.items():
            self.log(f"val/profitability/{name}", value, prog_bar=False)
        self._profitability_buffer = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["optimizer"]["lr"],
            weight_decay=self.hparams["optimizer"]["weight_decay"],
            betas=tuple(self.hparams["optimizer"]["betas"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class MarketDataModule(pl.LightningDataModule):
    """DataModule that loads SQLite data and produces sliding windows."""

    def __init__(self, data_config: DataConfig, trainer_config: TrainerConfig, *, seed: int = 42) -> None:
        super().__init__()
        self.data_config = data_config
        self.trainer_config = trainer_config
        self.seed = seed
        self.registry = FeatureRegistry()
        self.train_dataset: SlidingWindowDataset | None = None
        self.val_dataset: SlidingWindowDataset | None = None
        self.curriculum_scheduler: CurriculumScheduler | None = None
        self.current_curriculum: CurriculumParameters | None = None
        self._enriched_panel = None
        self._feature_columns: list[str] = []
        self._target_columns: list[str] = []
        self._persistent_workers: bool | None = None
        self._state_token_columns: list[str] = []
        self._regime_source_columns: list[str] = []
        self._market_state_metadata: MarketStateMetadata | None = None

    def setup(self, stage: str | None = None) -> None:
        source = SQLiteMarketSource(path=str(self.data_config.sqlite_path))
        dataset = SQLiteMarketDataset(
            source=source,
            symbol_universe=self.data_config.symbol_universe,
            indicators=self.data_config.indicators,
            alternative_data=self.data_config.alternative_data,
            resample_rule=self.data_config.resample_rule,
            tz_convert=self.data_config.tz_convert,
        )
        panel = dataset.as_panel()
        pipeline = self.registry.build_pipeline(self.data_config.feature_set)
        enriched = pipeline.transform_panel(panel)
        enriched = self._prepare_market_state(enriched)
        # Columns to exclude from features: token columns and original regime columns
        # (regime columns are represented via embeddings, so including them as features is redundant)
        excluded_columns = set(self._state_token_columns) | set(self._regime_source_columns)
        logger.debug("State token columns to exclude: %s", self._state_token_columns)
        logger.debug("Regime source columns to exclude: %s", self._regime_source_columns)
        if self.data_config.feature_set:
            requested = [f for f in self.data_config.feature_set if f in enriched.columns]
            numeric_cols = enriched[requested].select_dtypes(include=[np.number]).columns.tolist()
            # Exclude any column containing 'regime' (handles edge cases)
            available = [col for col in numeric_cols if col not in excluded_columns and 'regime' not in col.lower()]
        else:
            numeric_panel = enriched.select_dtypes(include=[np.number])
            all_numeric = list(numeric_panel.columns)
            # Exclude any column containing 'regime' (handles edge cases)
            available = [c for c in all_numeric if c not in excluded_columns and 'regime' not in c.lower()]
            logger.info("Total numeric columns: %d, excluded: %d, available: %d",
                       len(all_numeric), len(excluded_columns), len(available))
        self._enriched_panel = enriched
        self._feature_columns = available
        self._target_columns = list(self.data_config.target_columns)
        # Log regime-related columns for debugging
        regime_in_features = [c for c in available if 'regime' in c.lower()]
        if regime_in_features:
            logger.warning("Regime columns still in features (should be excluded): %s", regime_in_features)

        if self.data_config.curriculum is not None:
            self.curriculum_scheduler = CurriculumScheduler(self.data_config, self.data_config.curriculum)
            params = self.curriculum_scheduler.parameters_for_epoch(0)
        else:
            params = CurriculumParameters(
                window_size=self.data_config.window_size,
                horizon=self.data_config.horizon,
                stride=self.data_config.stride,
                normalise=self.data_config.normalise,
            )

        self.apply_curriculum(params)

    def _build_dataset(self, params: CurriculumParameters) -> SlidingWindowDataset:
        if self._enriched_panel is None:
            raise RuntimeError("DataModule.setup must be called before building datasets")
        logger.debug("Building dataset with %d feature columns and %d state columns",
                    len(self._feature_columns), len(self._state_token_columns))
        return SlidingWindowDataset(
            panel=self._enriched_panel,
            feature_columns=self._feature_columns,
            target_columns=self._target_columns,
            window_size=params.window_size,
            horizon=params.horizon,
            stride=params.stride,
            normalise=params.normalise,
            state_columns=self._state_token_columns,
        )

    def _assign_splits(self, dataset: SlidingWindowDataset) -> None:
        dataset_length = len(dataset)
        if dataset_length == 0:
            raise ValueError("SlidingWindowDataset is empty. Check window size, horizon, and data availability.")

        val_fraction = max(0.0, min(1.0, self.data_config.val_fraction))
        if val_fraction == 0.0 or dataset_length < 2:
            self.train_dataset = dataset
            self.val_dataset = dataset
            return

        val_len = int(round(dataset_length * val_fraction))
        if val_len <= 0:
            val_len = 1
        if val_len >= dataset_length:
            val_len = dataset_length - 1
        train_len = dataset_length - val_len

        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(dataset, lengths=[train_len, val_len], generator=generator)
        self.train_dataset = train_subset
        self.val_dataset = val_subset

    def apply_curriculum(self, params: CurriculumParameters) -> None:
        dataset = self._build_dataset(params)
        self._assign_splits(dataset)
        self.current_curriculum = params

    def step_curriculum(self, epoch: int) -> CurriculumParameters | None:
        if self.curriculum_scheduler is None:
            return None
        params = self.curriculum_scheduler.parameters_for_epoch(epoch)
        if self.current_curriculum == params:
            return None
        self.apply_curriculum(params)
        return params

    def _prepare_market_state(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._state_token_columns = []
        self._regime_source_columns = []  # Track original regime columns to exclude from features
        self._market_state_metadata = None
        regime_columns = [col for col in panel.columns if col.startswith("regime__") and not col.endswith("__token")]
        if not regime_columns:
            return panel

        enriched = panel.copy()
        metadata_columns: list[tuple[str, str, list[str]]] = []
        for column in sorted(regime_columns):
            base_name = column.split("regime__", 1)[1] or column
            token_column = f"{column}__token"
            filled = enriched[column].fillna("__missing__").astype(str)
            categorical = pd.Categorical(filled)
            enriched[token_column] = categorical.codes.astype(np.int64)
            self._state_token_columns.append(token_column)
            self._regime_source_columns.append(column)  # Track original column
            metadata_columns.append((base_name, token_column, categorical.categories.astype(str).tolist()))

        if metadata_columns:
            self._market_state_metadata = MarketStateMetadata.from_columns(metadata_columns)
        return enriched

    @property
    def has_curriculum(self) -> bool:
        return self.curriculum_scheduler is not None

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=True,
            num_workers=self.trainer_config.num_workers,
            pin_memory=self.trainer_config.accelerator != "cpu",
            persistent_workers=self._persistent_workers_enabled(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=False,
            num_workers=self.trainer_config.num_workers,
            pin_memory=self.trainer_config.accelerator != "cpu",
            persistent_workers=self._persistent_workers_enabled(),
        )

    def _persistent_workers_enabled(self) -> bool:
        if self._persistent_workers is not None:
            return self._persistent_workers

        requested = bool(
            self.trainer_config.persistent_workers and self.trainer_config.num_workers > 0
        )
        if requested and _IS_WINDOWS:
            logger.warning(
                "Persistent DataLoader workers are not supported on Windows; disabling to avoid hangs."
            )
            self._persistent_workers = False
            return False

        self._persistent_workers = requested
        return requested

    def dataset_summary(self) -> dict[str, int]:
        """Return a lightweight summary of prepared datasets for logging."""

        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError("DataModule.setup must run before requesting a dataset summary")

        batch_size = max(1, self.trainer_config.batch_size)
        train_len = len(self.train_dataset)
        val_len = len(self.val_dataset)
        return {
            "train_windows": train_len,
            "val_windows": val_len,
            "train_batches": math.ceil(train_len / batch_size),
            "val_batches": math.ceil(val_len / batch_size),
            "feature_dim": self.feature_dim,
            "market_state_features": self.market_state_feature_count,
        }

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def feature_dim(self) -> int:
        if not self._feature_columns:
            raise RuntimeError("DataModule.setup must populate feature columns before access")
        return len(self._feature_columns)

    @property
    def market_state_metadata(self) -> MarketStateMetadata | None:
        return self._market_state_metadata

    @property
    def market_state_feature_count(self) -> int:
        if self._market_state_metadata is None:
            return 0
        return self._market_state_metadata.feature_count()


def load_experiment_from_file(path: Path) -> ExperimentConfig:
    with path.open("r") as fp:
        raw = yaml.safe_load(fp)
    data_section = dict(raw["data"])
    data_section["sqlite_path"] = Path(data_section["sqlite_path"])
    if data_section.get("symbol_universe") is not None:
        data_section["symbol_universe"] = list(data_section["symbol_universe"])
    if data_section.get("feature_set") is not None:
        data_section["feature_set"] = list(data_section["feature_set"])
    if data_section.get("curriculum") is not None:
        curriculum_section = dict(data_section["curriculum"])
        stages_data = curriculum_section.get("stages") or []
        stages: list[CurriculumStage] = []
        for stage in stages_data:
            stage_dict = dict(stage)
            stages.append(
                CurriculumStage(
                    start_epoch=int(stage_dict["start_epoch"]),
                    window_size=stage_dict.get("window_size"),
                    horizon=stage_dict.get("horizon"),
                    stride=stage_dict.get("stride"),
                    normalise=stage_dict.get("normalise"),
                )
            )
        data_section["curriculum"] = CurriculumConfig(
            stages=stages,
            repeat_final=curriculum_section.get("repeat_final", True),
        )
    if data_section.get("alternative_data") is not None:
        alt_specs: list[AlternativeDataSpec] = []
        for entry in data_section["alternative_data"] or []:
            entry_dict = dict(entry)
            name = entry_dict["name"]
            table = entry_dict["table"]
            join_columns = tuple(entry_dict.get("join_columns") or ("timestamp", "symbol"))
            columns = entry_dict.get("columns")
            filters = {
                key: tuple(value)
                for key, value in (entry_dict.get("filters") or {}).items()
            }
            parse_dates = entry_dict.get("parse_dates")
            alt_specs.append(
                AlternativeDataSpec(
                    name=name,
                    table=table,
                    join_columns=join_columns,
                    columns=tuple(columns) if columns is not None else None,
                    prefix=entry_dict.get("prefix"),
                    fill_forward=entry_dict.get("fill_forward", True),
                    fill_backward=entry_dict.get("fill_backward", False),
                    filters=filters,
                    parse_dates=tuple(parse_dates) if parse_dates is not None else None,
                )
            )
        data_section["alternative_data"] = alt_specs
    else:
        data_section["alternative_data"] = []
    data_cfg = DataConfig(**data_section)

    model_section = dict(raw["model"])
    if "conv_dilations" in model_section:
        model_section["conv_dilations"] = tuple(model_section["conv_dilations"])
    else:
        model_section["conv_dilations"] = (1, 2, 4, 8, 16, 32)
    if "architecture" in model_section:
        model_section["architecture"] = str(model_section["architecture"]).lower()
    calibration_section = model_section.get("calibration")
    if calibration_section is not None:
        calib_section = dict(calibration_section)
        quantiles = calib_section.get("quantiles")
        if quantiles is not None:
            quantiles_tuple = tuple(float(q) for q in quantiles)
        else:
            quantiles_tuple = (0.05, 0.5, 0.95)
        model_section["calibration"] = CalibrationConfig(
            enabled=bool(calib_section.get("enabled", False)),
            quantiles=quantiles_tuple,
            temperature=float(calib_section.get("temperature", 1.0)),
            min_concentration=float(calib_section.get("min_concentration", 1e-2)),
        )
    else:
        model_section["calibration"] = CalibrationConfig()
    market_state_section = model_section.get("market_state")
    if market_state_section is not None:
        state_section = dict(market_state_section)
        include = state_section.get("include")
        include_tuple = (
            tuple(str(name) for name in include)
            if include is not None
            else tuple()
        )
        model_section["market_state"] = MarketStateConfig(
            enabled=bool(state_section.get("enabled", False)),
            embedding_dim=int(state_section.get("embedding_dim", 16)),
            dropout=float(state_section.get("dropout", 0.0)),
            include=include_tuple,
        )
    else:
        model_section["market_state"] = MarketStateConfig()
    model_cfg = ModelConfig(**model_section)
    optimizer_cfg = OptimizerConfig(**raw.get("optimizer", {}))
    trainer_section = dict(raw.get("trainer", {}))
    if "checkpoint_dir" in trainer_section:
        trainer_section["checkpoint_dir"] = Path(trainer_section["checkpoint_dir"])
    trainer_cfg = TrainerConfig(**trainer_section)

    guardrails_section = raw.get("guardrails")

    def _optional_float(section: dict[str, object], key: str) -> float | None:
        value = section.get(key)
        if value is None:
            return None
        return float(value)

    if guardrails_section is None:
        guardrail_cfg = GuardrailConfig()
    else:
        guardrail_dict = dict(guardrails_section)
        sector_caps = {
            str(name): float(limit)
            for name, limit in (guardrail_dict.get("sector_caps") or {}).items()
        }
        factor_caps = {
            str(name): float(limit)
            for name, limit in (guardrail_dict.get("factor_caps") or {}).items()
        }
        guardrail_cfg = GuardrailConfig(
            enabled=bool(guardrail_dict.get("enabled", False)),
            capital_base=float(guardrail_dict.get("capital_base", 1.0)),
            tail_percentile=float(guardrail_dict.get("tail_percentile", 5.0)),
            max_gross_exposure=_optional_float(guardrail_dict, "max_gross_exposure"),
            max_net_exposure=_optional_float(guardrail_dict, "max_net_exposure"),
            max_turnover=_optional_float(guardrail_dict, "max_turnover"),
            min_tail_return=_optional_float(guardrail_dict, "min_tail_return"),
            max_tail_frequency=_optional_float(guardrail_dict, "max_tail_frequency"),
            max_symbol_exposure=_optional_float(guardrail_dict, "max_symbol_exposure"),
            sector_caps=sector_caps,
            sector_column=str(guardrail_dict.get("sector_column", "sector")),
            factor_caps=factor_caps,
            factor_column=str(guardrail_dict.get("factor_column", "factor")),
            timestamp_col=str(guardrail_dict.get("timestamp_col", "timestamp")),
            symbol_col=str(guardrail_dict.get("symbol_col", "symbol")),
            notional_col=str(guardrail_dict.get("notional_col", "notional")),
            position_col=str(guardrail_dict.get("position_col", "position")),
            price_col=str(guardrail_dict.get("price_col", "price")),
            return_col=str(guardrail_dict.get("return_col", "pnl")),
            enforcement=str(guardrail_dict.get("enforcement", "clip")),
        )

    pretraining_cfg: PretrainingConfig | None = None
    if "pretraining" in raw:
        pretraining_section = dict(raw["pretraining"])
        if "augmentations" in pretraining_section and pretraining_section["augmentations"] is not None:
            pretraining_section["augmentations"] = tuple(
                str(name).lower() for name in pretraining_section["augmentations"]
            )
        if "objective" in pretraining_section and pretraining_section["objective"] is not None:
            pretraining_section["objective"] = str(pretraining_section["objective"]).lower()
        if "mask_value" in pretraining_section:
            pretraining_section["mask_value"] = pretraining_section["mask_value"]
        if "time_mask_fill" in pretraining_section:
            pretraining_section["time_mask_fill"] = pretraining_section["time_mask_fill"]
        pretraining_cfg = PretrainingConfig(**pretraining_section)
    reinforcement_cfg: ReinforcementConfig | None = None
    if "reinforcement" in raw:
        reinforcement_section = dict(raw["reinforcement"])
        if "costs" in reinforcement_section and reinforcement_section["costs"] is not None:
            costs_section = reinforcement_section["costs"]
            if isinstance(costs_section, dict):
                reinforcement_section["costs"] = TradingCosts(**costs_section)
        buffer_section = reinforcement_section.get("replay_buffer")
        if buffer_section is not None:
            buffer_dict = dict(buffer_section)
            reinforcement_section["replay_buffer"] = ReplayBufferConfig(
                enabled=bool(buffer_dict.get("enabled", False)),
                capacity=int(buffer_dict.get("capacity", 16384)),
                sample_ratio=float(buffer_dict.get("sample_ratio", 0.5)),
                min_samples=int(buffer_dict.get("min_samples", 2048)),
            )
        reinforcement_cfg = ReinforcementConfig(**reinforcement_section)
    diagnostics_section = raw.get("diagnostics")
    if diagnostics_section is None:
        diagnostics_cfg = DiagnosticsConfig()
    else:
        diag_section = dict(diagnostics_section)
        def _optional_float(key: str) -> float | None:
            value = diag_section.get(key)
            if value is None:
                return None
            return float(value)

        diagnostics_cfg = DiagnosticsConfig(
            enabled=bool(diag_section.get("enabled", True)),
            log_interval=int(diag_section.get("log_interval", 50)),
            profile=bool(diag_section.get("profile", False)),
            gradient_noise_threshold=_optional_float("gradient_noise_threshold"),
            calibration_bias_threshold=_optional_float("calibration_bias_threshold"),
            calibration_error_threshold=_optional_float("calibration_error_threshold"),
        )
    wandb_tags = raw.get("wandb_tags")
    if wandb_tags is None:
        tags_tuple: tuple[str, ...] = ()
    else:
        tags_tuple = tuple(str(tag) for tag in wandb_tags)
    return ExperimentConfig(
        seed=raw.get("seed", 42),
        data=data_cfg,
        model=model_cfg,
        optimizer=optimizer_cfg,
        guardrails=guardrail_cfg,
        trainer=trainer_cfg,
        wandb_project=raw.get("wandb_project"),
        wandb_entity=raw.get("wandb_entity"),
        wandb_run_name=raw.get("wandb_run_name"),
        wandb_tags=tags_tuple,
        wandb_offline=bool(raw.get("wandb_offline", False)),
        notes=raw.get("notes"),
        pretraining=pretraining_cfg,
        reinforcement=reinforcement_cfg,
        diagnostics=diagnostics_cfg,
    )


def ensure_feature_dim_alignment(
    config: ExperimentConfig, data_module: MarketDataModule
) -> None:
    """Synchronise the model feature dimension with the engineered features."""

    try:
        base_dim = data_module.feature_dim
    except RuntimeError:
        # Setup has not been called yet – build once to reveal the engineered columns.
        data_module.setup(stage="fit")
        base_dim = data_module.feature_dim

    if base_dim <= 0:
        raise ValueError("No features available after preprocessing; check the feature pipeline configuration.")

    extra_dim = 0
    metadata = data_module.market_state_metadata
    if config.model.market_state.enabled:
        if metadata is None or metadata.feature_count() == 0:
            logger.warning(
                "Market-state embeddings enabled but no categorical state metadata is available; proceeding without the extra inputs."
            )
        else:
            selected = metadata.select(config.model.market_state.include)
            if selected.feature_count() == 0:
                logger.warning(
                    "Requested market-state features %s are unavailable; proceeding without embeddings.",
                    config.model.market_state.include,
                )
            else:
                extra_dim = config.model.market_state.embedding_dim * selected.feature_count()

    inferred_dim = base_dim + extra_dim
    if config.model.feature_dim != inferred_dim:
        logger.info(
            "Adjusting model feature dimension from %s to %s (base=%s, market-state extra=%s)",
            config.model.feature_dim,
            inferred_dim,
            base_dim,
            extra_dim,
        )
        config.model.feature_dim = inferred_dim


def instantiate_modules(config: ExperimentConfig) -> tuple[MarketLightningModule, MarketDataModule]:
    pl.seed_everything(config.seed)
    data_module = MarketDataModule(config.data, config.trainer, seed=config.seed)
    ensure_feature_dim_alignment(config, data_module)
    if config.model.market_state.enabled and data_module.market_state_metadata is None:
        # ensure metadata is available when embeddings are requested
        data_module.setup(stage="fit")
    metadata = data_module.market_state_metadata
    selected_metadata: MarketStateMetadata | None = None
    if metadata is not None and config.model.market_state.enabled:
        selected = metadata.select(config.model.market_state.include)
        if selected.feature_count() > 0:
            selected_metadata = selected
    lightning_module = MarketLightningModule(
        config.model,
        config.optimizer,
        market_state_metadata=selected_metadata,
    )
    return lightning_module, data_module


def _normalise_logged_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert arbitrary metric scalars (tensors, numpy types) to floats."""

    normalised: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            if isinstance(value, torch.Tensor):
                scalar = value.detach().cpu()
                if scalar.numel() != 1:
                    raise ValueError(
                        f"Logged metric '{key}' is not a scalar tensor: shape={tuple(scalar.shape)}"
                    )
                normalised[key] = float(scalar.item())
            elif hasattr(value, "item") and not isinstance(value, (int, float)):
                normalised[key] = float(value.item())  # type: ignore[arg-type]
            else:
                normalised[key] = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unable to convert metric '{key}'={value!r} to float") from exc
    return normalised


def run_training(
    config: ExperimentConfig,
    *,
    pretrain_checkpoint_path: str | Path | None = None,
) -> TrainingRunResult:
    module, data_module = instantiate_modules(config)
    if pretrain_checkpoint_path is not None:
        try:
            first_param = next(module.backbone.parameters())
            device = first_param.device
        except StopIteration:  # pragma: no cover - defensive guard
            device = torch.device("cpu")
        load_backbone_from_checkpoint(
            module.backbone,
            pretrain_checkpoint_path,
            device=device,
        )
        logger.info(
            "Loaded backbone weights from pretraining checkpoint %s",
            pretrain_checkpoint_path,
        )
    if data_module.train_dataset is None or data_module.val_dataset is None:
        data_module.setup(stage="fit")
    summary = data_module.dataset_summary()
    logger.info(
        "Prepared %s training windows and %s validation windows (batch size %s → %s steps/epoch)",
        summary["train_windows"],
        summary["val_windows"],
        config.trainer.batch_size,
        summary["train_batches"],
    )
    logger.info("Engineered feature dimension: %s columns", summary["feature_dim"])
    trainable_params, total_params = _parameter_counts(module)
    logger.info(
        "Model parameters: %.2fM trainable / %.2fM total",
        trainable_params / 1e6,
        total_params / 1e6,
    )
    checkpoint_dir = Path(config.trainer.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor_metric = config.trainer.monitor_metric
    sanitized_monitor = monitor_metric.replace("/", "_")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"plus-ultra-{{epoch:03d}}-{sanitized_monitor}-{{{sanitized_monitor}:.4f}}",
        monitor=monitor_metric,
        mode=config.trainer.monitor_mode,
        save_top_k=config.trainer.save_top_k,
        auto_insert_metric_name=False,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    if config.data.curriculum is not None:
        initial_params = CurriculumScheduler(config.data, config.data.curriculum).parameters_for_epoch(0)
    else:
        initial_params = CurriculumParameters(
            window_size=config.data.window_size,
            horizon=config.data.horizon,
            stride=config.data.stride,
            normalise=config.data.normalise,
        )
    module.update_horizon(initial_params.horizon)

    callbacks: list[pl.Callback] = [checkpoint_callback, lr_monitor]
    if config.data.curriculum is not None:
        callbacks.append(CurriculumCallback())
    if config.diagnostics.enabled:
        callbacks.append(
            TrainingDiagnosticsCallback(
                log_interval=max(1, config.diagnostics.log_interval),
                profile=config.diagnostics.profile,
                thresholds=config.diagnostics.as_thresholds(),
            )
        )

    loggers: list[pl.loggers.logger.Logger] = []
    wandb_logger = maybe_create_wandb_logger(config, run_kind="train")
    if wandb_logger is not None:
        wandb_logger.watch(module, log="gradients", log_freq=100, log_graph=False)
        loggers.append(wandb_logger)

    trainer_logger: pl.loggers.logger.Logger | list[pl.loggers.logger.Logger] | bool
    if not loggers:
        trainer_logger = True
    elif len(loggers) == 1:
        trainer_logger = loggers[0]
    else:
        trainer_logger = loggers

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        precision=config.trainer.precision,
        log_every_n_steps=config.trainer.log_every_n_steps,
        default_root_dir=str(checkpoint_dir),
        callbacks=callbacks,
        logger=trainer_logger,
        deterministic=True,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
    )
    trainer.fit(module, datamodule=data_module)
    best_model_path = checkpoint_callback.best_model_path or ""
    metrics = _normalise_logged_metrics(trainer.logged_metrics)
    profitability_summary = module.latest_profitability_summary or {}
    profitability_reports: dict[str, str] = {}
    if profitability_summary:
        json_path = write_metrics_report(
            profitability_summary,
            checkpoint_dir / "profitability_summary.json",
            format_hint="json",
        )
        md_path = write_metrics_report(
            profitability_summary,
            checkpoint_dir / "profitability_summary.md",
            format_hint="md",
        )
        profitability_reports = {
            "json": str(json_path),
            "markdown": str(md_path),
        }
    return TrainingRunResult(
        best_model_path=str(best_model_path),
        logged_metrics=metrics,
        dataset_summary=dict(summary),
        profitability_summary=profitability_summary,
        profitability_reports=profitability_reports,
    )

