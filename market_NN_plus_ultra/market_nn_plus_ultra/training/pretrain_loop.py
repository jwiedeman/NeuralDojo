"""Self-supervised pretraining loop for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..models.omni_mixture import MarketOmniBackbone, OmniBackboneConfig
from ..models.temporal_fusion import TemporalFusionConfig, TemporalFusionTransformer
from ..models.temporal_transformer import TemporalBackbone, TemporalBackboneConfig
from .config import ExperimentConfig, ModelConfig, OptimizerConfig, PretrainingConfig
from .train_loop import MarketDataModule


class MaskedTimeSeriesLightningModule(pl.LightningModule):
    """Lightning module for masked time-series reconstruction pretraining."""

    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        pretraining_config: PretrainingConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "model": asdict(model_config),
                "optimizer": asdict(optimizer_config),
                "pretraining": asdict(pretraining_config),
            }
        )
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.pretraining_config = pretraining_config

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
            omni_config = OmniBackboneConfig(
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
            self.backbone = MarketOmniBackbone(omni_config)
        else:
            raise ValueError(f"Unknown architecture '{model_config.architecture}'")

        self.reconstruction_head = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.model_dim),
            torch.nn.Linear(model_config.model_dim, model_config.model_dim),
            torch.nn.GELU(),
            torch.nn.Linear(model_config.model_dim, model_config.feature_dim),
        )

    def forward(self, masked_inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(masked_inputs)
        return self.reconstruction_head(hidden)

    def _mask_inputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_prob = self.pretraining_config.mask_prob
        mask = torch.rand(features.shape[:2], device=features.device) < mask_prob
        mask = mask.unsqueeze(-1)
        masked = features.clone()
        mask_value = self.pretraining_config.mask_value
        if mask_value == "mean":
            fill_value = features.mean(dim=1, keepdim=True)
            masked = torch.where(mask, fill_value, masked)
        else:
            masked = masked.masked_fill(mask, float(mask_value))
        return masked, mask

    def _loss_fn(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.expand_as(targets)
        if mask_expanded.any():
            preds = predictions[mask_expanded]
            actuals = targets[mask_expanded]
        else:
            preds = predictions.reshape(-1, targets.size(-1))
            actuals = targets.reshape(-1, targets.size(-1))

        if self.pretraining_config.loss.lower() == "mae":
            return F.l1_loss(preds, actuals)
        return F.mse_loss(preds, actuals)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features = batch["features"]
        masked, mask = self._mask_inputs(features)
        recon = self(masked)
        loss = self._loss_fn(recon, features, mask)
        self.log("train/pretrain_loss", loss, prog_bar=True)
        self.log("train/mask_ratio", mask.float().mean())
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        features = batch["features"]
        masked, mask = self._mask_inputs(features)
        recon = self(masked)
        loss = self._loss_fn(recon, features, mask)
        self.log("val/pretrain_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            betas=self.optimizer_config.betas,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def instantiate_pretraining_module(config: ExperimentConfig) -> tuple[MaskedTimeSeriesLightningModule, MarketDataModule]:
    if config.pretraining is None:
        raise ValueError("ExperimentConfig.pretraining must be provided for pretraining runs")
    pl.seed_everything(config.seed)
    module = MaskedTimeSeriesLightningModule(config.model, config.optimizer, config.pretraining)
    data_module = MarketDataModule(config.data, config.trainer, seed=config.seed)
    return module, data_module


def run_pretraining(config: ExperimentConfig) -> dict[str, Any]:
    module, data_module = instantiate_pretraining_module(config)
    checkpoint_dir = config.trainer.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor_metric = config.pretraining.monitor_metric
    sanitized_monitor = monitor_metric.replace("/", "_")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"plus-ultra-pretrain-{{epoch:03d}}-{sanitized_monitor}-{{{sanitized_monitor}:.4f}}",
        monitor=monitor_metric,
        mode="min",
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


__all__ = [
    "MaskedTimeSeriesLightningModule",
    "instantiate_pretraining_module",
    "run_pretraining",
]
