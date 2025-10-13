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
from ..models.losses import CompositeTradingLoss
from .config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, TrainerConfig


class MarketLightningModule(pl.LightningModule):
    """Lightning module combining the backbone and policy head."""

    def __init__(self, model_config: ModelConfig, optimizer_config: OptimizerConfig) -> None:
        super().__init__()
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
        else:
            raise ValueError(f"Unknown architecture '{model_config.architecture}'")
        self.head = TemporalPolicyHead(model_config.model_dim, model_config.horizon, model_config.output_dim)
        self.loss_fn = CompositeTradingLoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        return self.head(hidden)

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
        loss = self.loss_fn(preds, targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self(batch["features"])
        loss = self.loss_fn(preds, batch["targets"])
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
        full_dataset = SlidingWindowDataset(
            panel=enriched,
            feature_columns=available,
            target_columns=self.data_config.target_columns,
            window_size=self.data_config.window_size,
            horizon=self.data_config.horizon,
            stride=self.data_config.stride,
            normalise=self.data_config.normalise,
        )
        dataset_length = len(full_dataset)
        if dataset_length == 0:
            raise ValueError("SlidingWindowDataset is empty. Check window size, horizon, and data availability.")

        val_fraction = max(0.0, min(1.0, self.data_config.val_fraction))
        if val_fraction == 0.0 or dataset_length < 2:
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset
            return

        val_len = int(round(dataset_length * val_fraction))
        if val_len <= 0:
            val_len = 1
        if val_len >= dataset_length:
            val_len = dataset_length - 1
        train_len = dataset_length - val_len

        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(full_dataset, lengths=[train_len, val_len], generator=generator)
        self.train_dataset = train_subset
        self.val_dataset = val_subset

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=True,
            num_workers=self.trainer_config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=False,
            num_workers=self.trainer_config.num_workers,
            pin_memory=True,
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
    return ExperimentConfig(
        seed=raw.get("seed", 42),
        data=data_cfg,
        model=model_cfg,
        optimizer=optimizer_cfg,
        trainer=trainer_cfg,
        wandb_project=raw.get("wandb_project"),
        notes=raw.get("notes"),
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
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        precision=config.trainer.precision,
        log_every_n_steps=config.trainer.log_every_n_steps,
        default_root_dir=str(checkpoint_dir),
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=True,
    )
    trainer.fit(module, datamodule=data_module)
    return {
        "best_model_path": checkpoint_callback.best_model_path,
        "logged_metrics": trainer.logged_metrics,
    }

