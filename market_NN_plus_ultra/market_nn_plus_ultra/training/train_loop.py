"""Training loop orchestrator."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, random_split

from ..data import FeaturePipeline, SQLiteMarketDataset, SlidingWindowDataset, WindowConfig
from ..models.losses import default_risk_loss
from ..models.temporal_transformer import TemporalBackbone, TemporalBackboneConfig
from .config import TrainingConfig


def _collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    features, targets = zip(*batch)
    return torch.stack(features), torch.stack(targets)


class Trainer:
    """High-level trainer coordinating data, model, and optimisation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = default_risk_loss()
        self.amp_enabled = config.mixed_precision and self.device.type == "cuda"
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.amp_enabled:
            self.scaler = torch.cuda.amp.GradScaler()

        dataset = SQLiteMarketDataset(
            database_path=config.data.database_path,
            indicators=config.data.indicators,
            asset_universe=config.data.asset_universe,
            feature_pipeline=FeaturePipeline.with_default_indicators(),
        )
        panel = dataset.load_joined_panel().fillna(method="ffill").dropna()

        window_config = WindowConfig(
            window_size=config.window_size,
            forecast_horizon=config.model.forecast_horizon,
            stride=config.window_stride,
            target_column=config.target_column,
            normalise=True,
        )
        sliding_dataset = SlidingWindowDataset(panel, window_config)
        self.target_idx = sliding_dataset.target_idx

        n_total = len(sliding_dataset)
        n_val = max(int(0.1 * n_total), 1)
        n_train = max(n_total - n_val, 1)
        self.train_dataset, self.val_dataset = random_split(sliding_dataset, [n_train, n_val])

        feature_dim = len(sliding_dataset.feature_columns)
        if config.model.input_size is None:
            config.model.input_size = feature_dim

        backbone_config = TemporalBackboneConfig(
            input_size=config.model.input_size,
            d_model=config.model.d_model,
            depth=config.model.depth,
            n_heads=config.model.n_heads,
            patch_size=config.model.patch_size,
            dropout=config.model.dropout,
            conv_kernel=config.model.conv_kernel,
            conv_dilations=config.model.conv_dilations,
            ffn_expansion=config.model.ffn_expansion,
            forecast_horizon=config.model.forecast_horizon,
            output_size=config.model.output_size,
        )

        self.model = TemporalBackbone(backbone_config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs, eta_min=config.optimizer.lr * 0.1
        )

        self.checkpoint_dir = config.checkpoint_dir
        if self.checkpoint_dir is not None:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def dataloaders(self) -> tuple[DataLoader, DataLoader]:
        return (
            DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=_collate,
            ),
            DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=_collate,
            ),
        )

    def _compute_loss(self, preds: torch.Tensor, targets: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        last_close = features[:, -1, self.target_idx]
        returns = (targets - last_close.unsqueeze(-1)) / (last_close.unsqueeze(-1).abs() + 1e-6)
        actions = torch.stack(
            [
                torch.relu(returns),
                torch.zeros_like(returns),
                torch.relu(-returns),
            ],
            dim=-1,
        )
        pnl = (actions * preds.softmax(dim=-1)).sum(dim=-1)
        return self.loss_fn(preds, actions, pnl)

    def train(self) -> None:
        train_loader, val_loader = self.dataloaders()
        amp_context = torch.cuda.amp.autocast if self.amp_enabled else nullcontext

        for epoch in range(self.config.num_epochs):
            self.model.train()
            running_loss = 0.0
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with amp_context():
                    preds = self.model(features)
                    loss = self._compute_loss(preds, targets, features)

                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                    self.optimizer.step()

                running_loss += loss.item() * features.size(0)

            self.scheduler.step()

            val_loss = self.evaluate(val_loader)
            avg_loss = running_loss / len(self.train_dataset)
            print(f"Epoch {epoch+1}/{self.config.num_epochs} - train_loss={avg_loss:.4f} - val_loss={val_loss:.4f}")

            if self.checkpoint_dir is not None:
                ckpt_path = Path(self.checkpoint_dir) / f"epoch_{epoch+1:03d}.pt"
                torch.save({"model_state": self.model.state_dict()}, ckpt_path)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                preds = self.model(features)
            loss = self._compute_loss(preds, targets, features)
            total_loss += loss.item() * features.size(0)
            total_items += features.size(0)
        return total_loss / max(total_items, 1)
