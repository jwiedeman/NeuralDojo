"""Temporal Fusion Transformer style backbone for multi-horizon forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class TemporalFusionConfig:
    """Configuration for :class:`TemporalFusionTransformer`."""

    feature_dim: int
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    horizon: int = 5
    max_seq_len: int = 4096


class VariableSelectionNetwork(nn.Module):
    """Softly select informative features with learnable attention weights."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_transform = nn.Linear(feature_dim, feature_dim)
        self.score_network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(self.score_network(x), dim=-1)
        transformed = self.feature_transform(x)
        return self.dropout(scores * transformed)


class GatedResidualNetwork(nn.Module):
    """Gated residual block used throughout the transformer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.gate = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Sigmoid())
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        gated = self.gate(x)
        x = gated * x
        return self.norm(x + residual)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer style backbone producing contextual embeddings."""

    def __init__(self, config: TemporalFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.variable_selection = VariableSelectionNetwork(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.input_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        self.encoder_gate = GatedResidualNetwork(config.hidden_dim, config.hidden_dim * 2, config.hidden_dim, config.dropout)
        self.decoder_gate = GatedResidualNetwork(config.hidden_dim, config.hidden_dim * 2, config.hidden_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_dim))
        self.future_context = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        b, t, _ = x.shape
        if t > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {t} exceeds maximum positional encoding {self.config.max_seq_len}"
            )
        selected = self.variable_selection(x)
        encoder_tokens = self.input_proj(selected)
        pos = self.positional_encoding[:, :t, :]
        encoder_tokens = self.dropout(encoder_tokens + pos)
        memory = self.encoder(encoder_tokens)
        memory = self.encoder_gate(memory)

        future_tokens = self.future_context[:, : self.config.horizon, :].expand(b, -1, -1)
        decoded = self.decoder(tgt=future_tokens, memory=memory)
        decoded = self.decoder_gate(decoded)
        combined = torch.cat([memory, decoded], dim=1)
        return self.output_norm(combined)


def initialise_temporal_fusion(feature_dim: int, **overrides: Any) -> TemporalFusionTransformer:
    """Instantiate and initialise a :class:`TemporalFusionTransformer`."""

    config_dict: dict[str, Any] = {"feature_dim": feature_dim}
    config_dict.update(overrides)
    config = TemporalFusionConfig(**config_dict)
    model = TemporalFusionTransformer(config)
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    return model


__all__ = [
    "TemporalFusionConfig",
    "TemporalFusionTransformer",
    "initialise_temporal_fusion",
]
