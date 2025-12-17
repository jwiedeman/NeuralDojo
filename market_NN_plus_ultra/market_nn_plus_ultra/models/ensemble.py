"""Ensemble model support for Market NN Plus Ultra.

This module provides ensemble wrappers that combine predictions from multiple
backbone architectures for more robust trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_transformer import HybridTemporalTransformer
from .temporal_fusion import TemporalFusionTransformer
from .omni_mixture import OmniMixtureBackbone
from .moe_transformer import MoETransformerBackbone
from .multi_scale import MultiScaleBackbone


ARCHITECTURE_MAP = {
    "hybrid": HybridTemporalTransformer,
    "temporal_transformer": HybridTemporalTransformer,
    "tft": TemporalFusionTransformer,
    "temporal_fusion": TemporalFusionTransformer,
    "omni": OmniMixtureBackbone,
    "omni_mixture": OmniMixtureBackbone,
    "moe": MoETransformerBackbone,
    "moe_transformer": MoETransformerBackbone,
    "multi_scale": MultiScaleBackbone,
}


@dataclass(slots=True)
class EnsembleMemberConfig:
    """Configuration for a single ensemble member."""

    architecture: str
    checkpoint_path: Optional[str] = None
    weight: float = 1.0
    feature_dim: int = 128
    model_dim: int = 256
    depth: int = 6
    heads: int = 8
    dropout: float = 0.1
    horizon: int = 5
    output_dim: int = 1


@dataclass(slots=True)
class EnsembleConfig:
    """Configuration for the full ensemble."""

    members: List[EnsembleMemberConfig]
    aggregation: Literal["mean", "weighted", "attention", "voting"] = "weighted"
    calibrate_weights: bool = True
    dropout: float = 0.1


class EnsembleMember(nn.Module):
    """Wrapper for a single ensemble member with its own head."""

    def __init__(self, config: EnsembleMemberConfig) -> None:
        super().__init__()
        self.config = config

        # Build backbone
        backbone_cls = ARCHITECTURE_MAP.get(config.architecture)
        if backbone_cls is None:
            raise ValueError(f"Unknown architecture: {config.architecture}")

        # Common kwargs for most architectures
        kwargs = {
            "feature_dim": config.feature_dim,
            "model_dim": config.model_dim,
            "depth": config.depth,
            "heads": config.heads,
            "dropout": config.dropout,
        }

        self.backbone = backbone_cls(**kwargs)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, config.horizon * config.output_dim),
        )

        self.horizon = config.horizon
        self.output_dim = config.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning predictions shaped [batch, horizon, output_dim]."""
        hidden = self.backbone(x)
        last_state = hidden[:, -1, :]
        out = self.head(last_state)
        return out.view(-1, self.horizon, self.output_dim)

    def load_checkpoint(self, path: str | Path, device: torch.device | str = "cpu") -> None:
        """Load weights from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Filter for backbone weights
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                backbone_state[key.replace("backbone.", "")] = value
            elif not key.startswith("head."):
                backbone_state[key] = value

        self.backbone.load_state_dict(backbone_state, strict=False)


class AttentionAggregator(nn.Module):
    """Learn to weight ensemble members via attention."""

    def __init__(self, num_members: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, member_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            member_outputs: [batch, num_members, horizon, output_dim]

        Returns:
            aggregated: [batch, horizon, output_dim]
            weights: [batch, num_members]
        """
        batch, num_members, horizon, out_dim = member_outputs.shape

        # Flatten horizon and output dim for attention
        flat = member_outputs.view(batch, num_members, -1)

        keys = self.key_proj(flat)
        values = self.value_proj(flat)
        query = self.query.expand(batch, -1, -1)

        # Attention scores
        scores = torch.bmm(query, keys.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted combination
        attended = torch.bmm(weights, values)
        out = self.out_proj(attended)
        out = out.view(batch, horizon, out_dim)

        return out, weights.squeeze(1)


class EnsembleModel(nn.Module):
    """Ensemble of multiple trading model architectures.

    Combines predictions from multiple backbone architectures using various
    aggregation strategies for more robust trading signals.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        super().__init__()
        self.config = config

        # Build ensemble members
        self.members = nn.ModuleList([
            EnsembleMember(member_config)
            for member_config in config.members
        ])

        # Initialize weights
        self.member_weights = nn.Parameter(
            torch.tensor([m.weight for m in config.members]),
            requires_grad=config.calibrate_weights,
        )

        # Attention aggregator if needed
        if config.aggregation == "attention":
            hidden_dim = config.members[0].horizon * config.members[0].output_dim
            self.attention_agg = AttentionAggregator(
                num_members=len(config.members),
                hidden_dim=hidden_dim,
                dropout=config.dropout,
            )
        else:
            self.attention_agg = None

        self.horizon = config.members[0].horizon
        self.output_dim = config.members[0].output_dim

    def forward(
        self,
        x: torch.Tensor,
        return_member_outputs: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members.

        Args:
            x: Input features [batch, seq_len, features]
            return_member_outputs: If True, also return individual member outputs

        Returns:
            aggregated: [batch, horizon, output_dim]
            (optional) member_outputs: [batch, num_members, horizon, output_dim]
            (optional) weights: [batch, num_members]
        """
        # Get outputs from all members
        member_outputs = torch.stack([
            member(x) for member in self.members
        ], dim=1)  # [batch, num_members, horizon, output_dim]

        # Aggregate based on strategy
        if self.config.aggregation == "mean":
            aggregated = member_outputs.mean(dim=1)
            weights = torch.ones(x.size(0), len(self.members), device=x.device) / len(self.members)

        elif self.config.aggregation == "weighted":
            # Normalize weights to sum to 1
            norm_weights = F.softmax(self.member_weights, dim=0)
            weights = norm_weights.unsqueeze(0).expand(x.size(0), -1)

            # Weighted average
            aggregated = (member_outputs * norm_weights.view(1, -1, 1, 1)).sum(dim=1)

        elif self.config.aggregation == "attention":
            aggregated, weights = self.attention_agg(member_outputs)

        elif self.config.aggregation == "voting":
            # For classification-like outputs, use voting
            # For regression, use median
            aggregated = member_outputs.median(dim=1).values
            weights = torch.ones(x.size(0), len(self.members), device=x.device) / len(self.members)

        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")

        if return_member_outputs:
            return aggregated, member_outputs, weights

        return aggregated

    def load_member_checkpoints(
        self,
        checkpoint_paths: Dict[int, str | Path],
        device: torch.device | str = "cpu",
    ) -> None:
        """Load checkpoints for specific ensemble members."""
        for idx, path in checkpoint_paths.items():
            if idx < len(self.members):
                self.members[idx].load_checkpoint(path, device)

    def get_member_weights(self) -> torch.Tensor:
        """Return normalized member weights."""
        if self.config.aggregation == "weighted":
            return F.softmax(self.member_weights, dim=0)
        return torch.ones(len(self.members)) / len(self.members)


def create_default_ensemble(
    feature_dim: int = 128,
    model_dim: int = 256,
    horizon: int = 5,
    output_dim: int = 1,
    device: torch.device | str = "cpu",
) -> EnsembleModel:
    """Create a default ensemble with diverse architectures."""
    members = [
        EnsembleMemberConfig(
            architecture="omni_mixture",
            weight=1.0,
            feature_dim=feature_dim,
            model_dim=model_dim,
            depth=6,
            heads=8,
            horizon=horizon,
            output_dim=output_dim,
        ),
        EnsembleMemberConfig(
            architecture="hybrid",
            weight=0.8,
            feature_dim=feature_dim,
            model_dim=model_dim,
            depth=6,
            heads=8,
            horizon=horizon,
            output_dim=output_dim,
        ),
        EnsembleMemberConfig(
            architecture="moe",
            weight=0.6,
            feature_dim=feature_dim,
            model_dim=model_dim,
            depth=4,
            heads=8,
            horizon=horizon,
            output_dim=output_dim,
        ),
    ]

    config = EnsembleConfig(
        members=members,
        aggregation="weighted",
        calibrate_weights=True,
    )

    model = EnsembleModel(config)
    return model.to(device)


class EnsembleTrainer:
    """Helper for training ensemble models."""

    def __init__(
        self,
        ensemble: EnsembleModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.ensemble = ensemble
        self.device = torch.device(device)
        self.ensemble.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                ensemble.parameters(),
                lr=1e-4,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

    def train_step(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        self.ensemble.train()

        features = features.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()

        predictions, member_outputs, weights = self.ensemble(
            features, return_member_outputs=True
        )

        # Main loss
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        main_loss = loss_fn(predictions, targets)

        # Optional: add diversity loss to encourage different predictions
        if member_outputs.size(1) > 1:
            # Pairwise correlation penalty
            flat_outputs = member_outputs.view(member_outputs.size(0), member_outputs.size(1), -1)
            mean_output = flat_outputs.mean(dim=1, keepdim=True)
            centered = flat_outputs - mean_output
            diversity_loss = -centered.var(dim=1).mean() * 0.01
        else:
            diversity_loss = torch.tensor(0.0, device=self.device)

        total_loss = main_loss + diversity_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "main_loss": main_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "weights": weights[0].detach().cpu().tolist(),
        }


__all__ = [
    "EnsembleConfig",
    "EnsembleMemberConfig",
    "EnsembleModel",
    "EnsembleTrainer",
    "create_default_ensemble",
]
