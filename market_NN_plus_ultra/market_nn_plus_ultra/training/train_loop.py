"""Training loop orchestrator."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.sqlite_loader import SQLiteMarketDataset
from ..models.temporal_transformer import TemporalBackbone, TemporalBackboneConfig
from ..models.losses import default_risk_loss
from .config import TrainingConfig


class MarketDataset(Dataset):
    """Placeholder torch dataset wrapping pandas panels."""

    def __init__(self, panel):
        self.panel = panel
        self.asset_ids = panel.index.get_level_values(0).unique()

    def __len__(self) -> int:
        return len(self.asset_ids)

    def __getitem__(self, idx: int):
        asset_id = self.asset_ids[idx]
        history = self.panel.xs(asset_id)
        features = torch.tensor(history.values, dtype=torch.float32)
        return features[:-1], features[1:, 0]  # simplistic target placeholder


class Trainer:
    """High-level trainer coordinating data, model, and optimisation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = default_risk_loss()

        dataset = SQLiteMarketDataset(
            database_path=config.data.database_path,
            indicators=config.data.indicators,
            asset_universe=config.data.asset_universe,
        )
        panel = dataset.load_joined_panel()
        feature_dim = panel.shape[1]
        self.model = TemporalBackbone(
            TemporalBackboneConfig(
                input_size=feature_dim,
                d_model=config.model.d_model,
                depth=config.model.depth,
                n_heads=config.model.n_heads,
                dilation_rates=config.model.dilation_rates,
                dropout=config.model.dropout,
                forecast_horizon=config.model.forecast_horizon,
                output_size=config.model.output_size,
            )
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        self.panel = panel

    def dataloader(self) -> DataLoader:
        dataset = MarketDataset(self.panel)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

    def train(self) -> None:
        self.model.train()
        loader = self.dataloader()
        for epoch in range(self.config.num_epochs):
            for batch_idx, (features, targets) in enumerate(loader):
                features = features.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(features)
                pnl = preds.mean(dim=-1)  # placeholder PnL estimate
                loss = self.loss_fn(preds, targets.unsqueeze(-1), pnl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad()
            # TODO: log metrics, run validation, save checkpoints
