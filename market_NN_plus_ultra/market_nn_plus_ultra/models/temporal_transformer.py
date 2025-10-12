"""Hybrid temporal transformer for market modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange
from torch import nn


@dataclass(slots=True)
class TemporalBackboneConfig:
    feature_dim: int
    model_dim: int = 512
    depth: int = 16
    heads: int = 8
    dropout: float = 0.1
    conv_kernel_size: int = 5
    conv_dilations: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
    )


class PatchEmbedding(nn.Module):
    def __init__(self, feature_dim: int, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TemporalConvMixer(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = rearrange(x, "b t d -> b d t")
        x = self.net(x)
        x = rearrange(x, "b d t -> b t d")
        return self.norm(x + residual)


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.conv = TemporalConvMixer(dim, kernel_size=kernel_size, dilation=dilation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(attn_out + residual)
        x = x + self.ff(x)
        x = self.conv(x)
        return x


class TemporalBackbone(nn.Module):
    """Deep hybrid transformer backbone for market time-series."""

    def __init__(self, config: TemporalBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = PatchEmbedding(config.feature_dim, config.model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 4096, config.model_dim))
        self.dropout = nn.Dropout(config.dropout)
        blocks = []
        for i in range(config.depth):
            dilation = config.conv_dilations[i % len(config.conv_dilations)]
            blocks.append(
                TemporalBlock(
                    dim=config.model_dim,
                    heads=config.heads,
                    dropout=config.dropout,
                    kernel_size=config.conv_kernel_size,
                    dilation=dilation,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        b, t, _ = x.shape
        if t > self.positional_encoding.shape[1]:
            raise ValueError(
                f"Sequence length {t} exceeds maximum positional encoding {self.positional_encoding.shape[1]}"
            )
        x = self.embed(x)
        pos = self.positional_encoding[:, :t, :]
        x = x + pos
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TemporalPolicyHead(nn.Module):
    """Policy/value projection head for trading decisions or forecasts."""

    def __init__(self, model_dim: int, horizon: int, output_dim: int = 3) -> None:
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, horizon * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, dim], we use last timestep representation
        last_state = x[:, -1, :]
        logits = self.proj(last_state)
        return logits.view(x.size(0), self.horizon, self.output_dim)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def initialise_backbone(feature_dim: int, **overrides: Any) -> TemporalBackbone:
    config_dict = {"feature_dim": feature_dim}
    config_dict.update(overrides)
    config = TemporalBackboneConfig(**config_dict)
    backbone = TemporalBackbone(config)
    for name, param in backbone.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    return backbone

