"""Self-supervised pretraining loop for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any
import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ..models.omni_mixture import MarketOmniBackbone, OmniBackboneConfig
from ..models.multi_scale import MultiScaleBackbone, MultiScaleBackboneConfig
from ..models.temporal_fusion import TemporalFusionConfig, TemporalFusionTransformer
from ..models.temporal_transformer import TemporalBackbone, TemporalBackboneConfig
from ..models.moe_transformer import MixtureOfExpertsBackbone, MixtureOfExpertsConfig
from .config import ExperimentConfig, ModelConfig, OptimizerConfig, PretrainingConfig
from ..utils.wandb import maybe_create_wandb_logger
from .train_loop import MarketDataModule, ensure_feature_dim_alignment, _parameter_counts


logger = logging.getLogger(__name__)


def _build_backbone(model_config: ModelConfig) -> nn.Module:
    """Return the configured backbone for a given model configuration."""

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
        return TemporalBackbone(backbone_config)
    if architecture in {"temporal_fusion", "tft"}:
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
        return TemporalFusionTransformer(fusion_config)
    if architecture in {"moe", "moe_transformer", "mixture_of_experts"}:
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
        return MixtureOfExpertsBackbone(moe_config)
    if architecture in {"omni", "omni_mixture", "omni_backbone"}:
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
            gradient_checkpointing=model_config.gradient_checkpointing,
        )
        return MarketOmniBackbone(omni_config)
    if architecture in {"multi_scale", "multiscale", "hierarchical"}:
        multi_config = MultiScaleBackboneConfig(
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
        return MultiScaleBackbone(multi_config)
    raise ValueError(f"Unknown architecture '{model_config.architecture}'")


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

        self.backbone = _build_backbone(model_config)

        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(model_config.model_dim),
            nn.Linear(model_config.model_dim, model_config.model_dim),
            nn.GELU(),
            nn.Linear(model_config.model_dim, model_config.feature_dim),
        )

    def forward(self, masked_inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(masked_inputs)
        return self.reconstruction_head(hidden)

    def _mask_inputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_prob = self.pretraining_config.mask_prob
        mask = torch.rand(features.shape[:2], device=features.device) < mask_prob
        mask = mask.unsqueeze(-1)
        mask_expanded = mask.expand_as(features)
        mask_value = self.pretraining_config.mask_value
        if mask_value == "mean":
            fill_value = features.mean(dim=1, keepdim=True).expand_as(features)
            masked = torch.where(mask_expanded, fill_value, features)
        else:
            masked = features.masked_fill(mask_expanded, float(mask_value))
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
        if getattr(self, "_trainer", None) is not None:
            self.log("train/pretrain_loss", loss, prog_bar=True)
            self.log("train/mask_ratio", mask.float().mean())
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        features = batch["features"]
        masked, mask = self._mask_inputs(features)
        recon = self(masked)
        loss = self._loss_fn(recon, features, mask)
        if getattr(self, "_trainer", None) is not None:
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


class ContrastiveTimeSeriesLightningModule(pl.LightningModule):
    """Contrastive self-supervised objective inspired by TS2Vec."""

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

        self.backbone = _build_backbone(model_config)
        projection_dim = pretraining_config.projection_dim
        self.projection_head = nn.Sequential(
            nn.LayerNorm(model_config.model_dim),
            nn.Linear(model_config.model_dim, model_config.model_dim),
            nn.GELU(),
            nn.Linear(model_config.model_dim, projection_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(inputs)
        pooled = hidden.mean(dim=1)
        return self.projection_head(pooled)

    # ------------------------------------------------------------------
    # augmentations
    # ------------------------------------------------------------------
    def _apply_jitter(self, tensor: torch.Tensor) -> torch.Tensor:
        std = float(self.pretraining_config.jitter_std)
        if std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * std
        return tensor + noise

    def _apply_scaling(self, tensor: torch.Tensor) -> torch.Tensor:
        std = float(self.pretraining_config.scaling_std)
        if std <= 0:
            return tensor
        scale = torch.randn(tensor.size(0), 1, 1, device=tensor.device, dtype=tensor.dtype) * std + 1.0
        return tensor * scale

    def _apply_time_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        ratio = float(self.pretraining_config.time_mask_ratio)
        if ratio <= 0 or tensor.size(1) == 0:
            return tensor
        length = max(1, int(round(tensor.size(1) * ratio)))
        if length >= tensor.size(1):
            length = tensor.size(1)
        starts = torch.randint(0, tensor.size(1) - length + 1, (tensor.size(0),), device=tensor.device)
        time_indices = torch.arange(tensor.size(1), device=tensor.device).unsqueeze(0)
        mask = (time_indices < starts.unsqueeze(1)) | (time_indices >= (starts + length).unsqueeze(1))
        mask = mask.unsqueeze(-1)
        fill_value = self.pretraining_config.time_mask_fill
        if isinstance(fill_value, str) and fill_value == "mean":
            fill = tensor.mean(dim=1, keepdim=True)
        else:
            fill = torch.full_like(tensor, float(fill_value))
        return torch.where(mask, tensor, fill)

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        augmented = tensor
        for name in self.pretraining_config.augmentations:
            if name == "jitter":
                augmented = self._apply_jitter(augmented)
            elif name == "scaling":
                augmented = self._apply_scaling(augmented)
            elif name == "time_mask":
                augmented = self._apply_time_mask(augmented)
        return augmented

    # ------------------------------------------------------------------
    # objective
    # ------------------------------------------------------------------
    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        temperature = float(self.pretraining_config.temperature)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.T) / max(temperature, 1e-6)
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_a + loss_b)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features = batch["features"]
        view_a = self._augment(features)
        view_b = self._augment(features)
        proj_a = self(view_a)
        proj_b = self(view_b)
        loss = self._info_nce(proj_a, proj_b)
        similarity = F.cosine_similarity(proj_a, proj_b, dim=-1).mean()
        if getattr(self, "_trainer", None) is not None:
            self.log("train/pretrain_loss", loss, prog_bar=True)
            self.log("train/contrastive_similarity", similarity)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        features = batch["features"]
        view_a = self._augment(features)
        view_b = self._augment(features)
        proj_a = self(view_a)
        proj_b = self(view_b)
        loss = self._info_nce(proj_a, proj_b)
        similarity = F.cosine_similarity(proj_a, proj_b, dim=-1).mean()
        if getattr(self, "_trainer", None) is not None:
            self.log("val/pretrain_loss", loss, prog_bar=True)
            self.log("val/contrastive_similarity", similarity)

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


def instantiate_pretraining_module(
    config: ExperimentConfig,
) -> tuple[pl.LightningModule, MarketDataModule]:
    if config.pretraining is None:
        raise ValueError("ExperimentConfig.pretraining must be provided for pretraining runs")
    pl.seed_everything(config.seed)
    objective = config.pretraining.objective.lower()
    data_module = MarketDataModule(config.data, config.trainer, seed=config.seed)
    ensure_feature_dim_alignment(config, data_module)

    if objective in {"masked", "mask"}:
        module: pl.LightningModule = MaskedTimeSeriesLightningModule(
            config.model, config.optimizer, config.pretraining
        )
    elif objective in {"contrastive", "ts2vec"}:
        module = ContrastiveTimeSeriesLightningModule(
            config.model, config.optimizer, config.pretraining
        )
    else:
        raise ValueError(f"Unknown pretraining objective '{config.pretraining.objective}'")
    return module, data_module


def _ensure_supported_accelerator(config: ExperimentConfig) -> None:
    """Downgrade to CPU if the requested accelerator isn't available."""

    accelerator = config.trainer.accelerator
    if accelerator is None:
        return

    normalized = accelerator.lower()
    if normalized in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return

        extra_help: list[str] = []
        cuda_version = getattr(torch.version, "cuda", None)
        if not cuda_version:
            # Installed PyTorch build does not include CUDA support.
            base_version = torch.__version__.split("+")[0]
            extra_help.append(
                "The current PyTorch installation was built without CUDA support. "
                "Install a CUDA-enabled wheel to train on the GPU."
            )
            extra_help.append(
                "For example: 'pip install --index-url https://download.pytorch.org/whl/cu121 "
                f"torch=={base_version}'"
            )
        else:
            extra_help.append(
                "PyTorch reports CUDA %s support but no GPU devices are available. "
                "Verify that the NVIDIA drivers are installed and CUDA_VISIBLE_DEVICES allows access."%
                cuda_version
            )

        help_message = " ".join(extra_help)
        message = (
            "Trainer accelerator set to '%s' but CUDA is not available. Falling back to CPU. %s"
            % (accelerator, help_message)
        )
        warnings.warn(message.strip(), RuntimeWarning, stacklevel=3)
        config.trainer.accelerator = "cpu"
        return

    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return
        warnings.warn(
            "Trainer accelerator set to 'mps' but the MPS backend is unavailable. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        config.trainer.accelerator = "cpu"


def run_pretraining(config: ExperimentConfig) -> dict[str, Any]:
    _ensure_supported_accelerator(config)
    matmul_precision = config.trainer.matmul_precision
    if matmul_precision:
        try:
            torch.set_float32_matmul_precision(matmul_precision)
            logger.info("Set float32 matmul precision to '%s'", matmul_precision)
        except (TypeError, ValueError) as exc:
            warnings.warn(
                f"Invalid matmul precision '{matmul_precision}' requested: {exc}. Skipping configuration.",
                RuntimeWarning,
                stacklevel=2,
            )
    module, data_module = instantiate_pretraining_module(config)
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
    loggers: list[pl.loggers.logger.Logger] = []
    wandb_logger = maybe_create_wandb_logger(config, run_kind="pretrain")
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
        callbacks=[checkpoint_callback, lr_monitor],
        logger=trainer_logger,
        deterministic=True,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
    )
    trainer.fit(module, datamodule=data_module)
    return {
        "best_model_path": checkpoint_callback.best_model_path,
        "logged_metrics": trainer.logged_metrics,
    }


__all__ = [
    "MaskedTimeSeriesLightningModule",
    "ContrastiveTimeSeriesLightningModule",
    "instantiate_pretraining_module",
    "run_pretraining",
]
