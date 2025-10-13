"""Mixture-of-Experts transformer backbone for ultra-deep trading models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .temporal_transformer import TemporalConvMixer


@dataclass(slots=True)
class MixtureOfExpertsConfig:
    """Configuration for :class:`MixtureOfExpertsBackbone`."""

    feature_dim: int
    model_dim: int = 768
    depth: int = 24
    heads: int = 12
    dropout: float = 0.1
    num_experts: int = 8
    ff_mult: int = 4
    router_dropout: float = 0.0
    conv_kernel_size: int = 7
    conv_dilations: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    )
    max_seq_len: int = 4096


class PatchEmbedding(nn.Module):
    """Linear projection that lifts feature vectors into the model dimension."""

    def __init__(self, feature_dim: int, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ExpertFeedForward(nn.Module):
    """One expert used inside the mixture-of-experts feed-forward block."""

    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseMixtureOfExperts(nn.Module):
    """Soft routing mixture-of-experts layer with residual connections."""

    def __init__(self, dim: int, num_experts: int, mult: int, dropout: float, router_dropout: float) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.router = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_experts),
        )
        self.router_dropout = nn.Dropout(router_dropout)
        self.experts = nn.ModuleList([ExpertFeedForward(dim, mult, dropout) for _ in range(num_experts)])
        self.output_norm = nn.LayerNorm(dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        gates = torch.softmax(self.router_dropout(logits), dim=-1)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        stacked = torch.stack(expert_outputs, dim=-2)  # [B, T, E, D]
        weighted = (gates.unsqueeze(-1) * stacked).sum(dim=-2)
        return self.output_norm(x + self.output_dropout(weighted))


class MixtureOfExpertsBlock(nn.Module):
    """Attention + mixture-of-experts feed-forward + temporal convolutions."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        num_experts: int,
        mult: int,
        router_dropout: float,
        conv_kernel: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.moe = DenseMixtureOfExperts(dim, num_experts, mult, dropout, router_dropout)
        self.conv = TemporalConvMixer(dim, kernel_size=conv_kernel, dilation=dilation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(attn_out + residual)
        x = self.moe(x)
        x = self.conv(x)
        return x


class MixtureOfExpertsBackbone(nn.Module):
    """Large-scale backbone that introduces mixture-of-experts capacity."""

    def __init__(self, config: MixtureOfExpertsConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = PatchEmbedding(config.feature_dim, config.model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.model_dim))
        self.dropout = nn.Dropout(config.dropout)

        blocks = []
        for layer_idx in range(config.depth):
            dilation = config.conv_dilations[layer_idx % len(config.conv_dilations)]
            blocks.append(
                MixtureOfExpertsBlock(
                    dim=config.model_dim,
                    heads=config.heads,
                    dropout=config.dropout,
                    num_experts=config.num_experts,
                    mult=config.ff_mult,
                    router_dropout=config.router_dropout,
                    conv_kernel=config.conv_kernel_size,
                    dilation=dilation,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        if length > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {length} exceeds maximum positional encoding {self.config.max_seq_len}"
            )
        x = self.embed(x)
        pos = self.positional_encoding[:, :length, :]
        x = self.dropout(x + pos)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def initialise_moe_backbone(feature_dim: int, **overrides: Any) -> MixtureOfExpertsBackbone:
    """Instantiate and initialise a :class:`MixtureOfExpertsBackbone`."""

    config_dict: dict[str, Any] = {"feature_dim": feature_dim}
    config_dict.update(overrides)
    config = MixtureOfExpertsConfig(**config_dict)
    backbone = MixtureOfExpertsBackbone(config)
    for name, param in backbone.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    return backbone


__all__ = [
    "MixtureOfExpertsBackbone",
    "MixtureOfExpertsConfig",
    "initialise_moe_backbone",
]
