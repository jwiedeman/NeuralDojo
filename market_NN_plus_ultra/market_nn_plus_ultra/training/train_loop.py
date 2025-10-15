"""Lightning-powered training loop for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..data import FeatureRegistry, SQLiteMarketDataset, SlidingWindowDataset
from ..data.sqlite_loader import SQLiteMarketSource
from ..models.temporal_transformer import TemporalBackbone, TemporalBackboneConfig, TemporalPolicyHead
from ..models.temporal_fusion import TemporalFusionConfig, TemporalFusionTransformer
from ..models.omni_mixture import MarketOmniBackbone, OmniBackboneConfig
from ..models.moe_transformer import MixtureOfExpertsBackbone, MixtureOfExpertsConfig
from ..models.state_space import StateSpaceBackbone, StateSpaceConfig
from ..models.losses import CompositeTradingLoss
from ..trading.pnl import TradingCosts
from ..utils.wandb import maybe_create_wandb_logger
from .config import (
    CurriculumConfig,
    CurriculumStage,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainingConfig,
    ReinforcementConfig,
    TrainerConfig,
)
from .curriculum import (
    CurriculumCallback,
    CurriculumParameters,
    CurriculumScheduler,
)


class MarketLightningModule(pl.LightningModule):
    """Lightning module combining the backbone and policy head."""

    def __init__(self, model_config: ModelConfig, optimizer_config: OptimizerConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.save_hyperparameters({"model": asdict(model_config), "optimizer": asdict(optimizer_config)})
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
            )
            self.backbone = MarketOmniBackbone(backbone_config)
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
        self.head = TemporalPolicyHead(model_config.model_dim, model_config.horizon, model_config.output_dim)
        self.loss_fn = CompositeTradingLoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        return self.head(hidden)

    def update_horizon(self, horizon: int) -> None:
        """Refresh the policy head when the training horizon changes."""

        if horizon == self.model_config.horizon:
            return
        self.model_config.horizon = horizon
        self.hparams["model"]["horizon"] = horizon
        new_head = TemporalPolicyHead(
            self.model_config.model_dim,
            horizon,
            self.model_config.output_dim,
        )
        self.head = new_head.to(self.device)

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
        preds = self(batch["features"])
        targets = batch["targets"]
        reference = batch.get("reference")
        loss = self.loss_fn(preds, targets, reference=reference)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self(batch["features"])
        reference = batch.get("reference")
        loss = self.loss_fn(preds, batch["targets"], reference=reference)
        self.log("val/loss", loss, prog_bar=True)

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

    def setup(self, stage: str | None = None) -> None:
        source = SQLiteMarketSource(path=str(self.data_config.sqlite_path))
        dataset = SQLiteMarketDataset(
            source=source,
            symbol_universe=self.data_config.symbol_universe,
            indicators=self.data_config.indicators,
            resample_rule=self.data_config.resample_rule,
            tz_convert=self.data_config.tz_convert,
        )
        panel = dataset.as_panel()
        pipeline = self.registry.build_pipeline(self.data_config.feature_set)
        enriched = pipeline.transform_panel(panel)
        if self.data_config.feature_set:
            available = [f for f in self.data_config.feature_set if f in enriched.columns]
        else:
            available = [c for c in enriched.columns if c not in ("symbol",)]
        self._enriched_panel = enriched
        self._feature_columns = available
        self._target_columns = list(self.data_config.target_columns)

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
        return SlidingWindowDataset(
            panel=self._enriched_panel,
            feature_columns=self._feature_columns,
            target_columns=self._target_columns,
            window_size=params.window_size,
            horizon=params.horizon,
            stride=params.stride,
            normalise=params.normalise,
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
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=False,
            num_workers=self.trainer_config.num_workers,
            pin_memory=self.trainer_config.accelerator != "cpu",
        )


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
    data_cfg = DataConfig(**data_section)

    model_section = dict(raw["model"])
    if "conv_dilations" in model_section:
        model_section["conv_dilations"] = tuple(model_section["conv_dilations"])
    else:
        model_section["conv_dilations"] = (1, 2, 4, 8, 16, 32)
    if "architecture" in model_section:
        model_section["architecture"] = str(model_section["architecture"]).lower()
    model_cfg = ModelConfig(**model_section)
    optimizer_cfg = OptimizerConfig(**raw.get("optimizer", {}))
    trainer_section = dict(raw.get("trainer", {}))
    if "checkpoint_dir" in trainer_section:
        trainer_section["checkpoint_dir"] = Path(trainer_section["checkpoint_dir"])
    trainer_cfg = TrainerConfig(**trainer_section)

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
        reinforcement_cfg = ReinforcementConfig(**reinforcement_section)
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
        trainer=trainer_cfg,
        wandb_project=raw.get("wandb_project"),
        wandb_entity=raw.get("wandb_entity"),
        wandb_run_name=raw.get("wandb_run_name"),
        wandb_tags=tags_tuple,
        wandb_offline=bool(raw.get("wandb_offline", False)),
        notes=raw.get("notes"),
        pretraining=pretraining_cfg,
        reinforcement=reinforcement_cfg,
    )


def instantiate_modules(config: ExperimentConfig) -> tuple[MarketLightningModule, MarketDataModule]:
    pl.seed_everything(config.seed)
    lightning_module = MarketLightningModule(config.model, config.optimizer)
    data_module = MarketDataModule(config.data, config.trainer, seed=config.seed)
    return lightning_module, data_module


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    module, data_module = instantiate_modules(config)
    checkpoint_dir = config.trainer.checkpoint_dir
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
    )
    trainer.fit(module, datamodule=data_module)
    return {
        "best_model_path": checkpoint_callback.best_model_path,
        "logged_metrics": trainer.logged_metrics,
    }

