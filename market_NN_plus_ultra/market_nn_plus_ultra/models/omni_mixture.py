"""Omni-scale backbone mixing attention, state-space mixers, and convolutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .temporal_transformer import TemporalConvMixer


@dataclass(slots=True)
class OmniBackboneConfig:
    """Configuration for :class:`MarketOmniBackbone`."""

    feature_dim: int
    model_dim: int = 768
    depth: int = 24
    heads: int = 12
    dropout: float = 0.1
    ff_mult: int = 4
    ssm_state_dim: int = 256
    ssm_kernel_size: int = 16
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
    coarse_factor: int = 4
    cross_every: int = 2
    max_seq_len: int = 4096


class StateSpaceMixer(nn.Module):
    """Depthwise state-space inspired mixer with gated convolutions."""

    def __init__(self, dim: int, state_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(dim, state_dim * 2)
        self.conv = nn.Conv1d(
            state_dim,
            state_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=state_dim,
        )
        self.output_proj = nn.Linear(state_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        mixed, gate = self.input_proj(x).chunk(2, dim=-1)
        mixed = mixed * torch.sigmoid(gate)
        mixed = rearrange(mixed, "b t d -> b d t")
        mixed = self.conv(mixed)[..., : x.size(1)]
        mixed = rearrange(mixed, "b d t -> b t d")
        mixed = self.output_proj(mixed)
        return self.norm(residual + self.dropout(mixed))


class CrossScaleAttention(nn.Module):
    """Attend from fine resolution tokens into coarse context."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(fine, coarse, coarse)
        return self.norm(fine + attn_out)


class FeedForward(nn.Module):
    """Gated feed-forward network with GELU activations."""

    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class OmniBlock(nn.Module):
    """One omni-scale processing block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        ff_mult: int,
        state_dim: int,
        ssm_kernel: int,
        conv_kernel: int,
        dilation: int,
        enable_cross: bool,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_mult, dropout)
        self.state_mixer = StateSpaceMixer(dim, state_dim, ssm_kernel, dropout)
        self.conv = TemporalConvMixer(dim, kernel_size=conv_kernel, dilation=dilation, dropout=dropout)
        self.cross = CrossScaleAttention(dim, heads, dropout) if enable_cross else None

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor | None) -> torch.Tensor:
        residual = fine
        attn_out, _ = self.attn(fine, fine, fine)
        fine = self.attn_norm(attn_out + residual)
        fine = self.ff(fine)
        fine = self.state_mixer(fine)
        fine = self.conv(fine)
        if self.cross is not None and coarse is not None:
            fine = self.cross(fine, coarse)
        return fine


class MarketOmniBackbone(nn.Module):
    """Large-scale market model blending attention, convolutions, and state-space mixers."""

    def __init__(self, config: OmniBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Linear(config.feature_dim, config.model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.model_dim))
        self.coarse_positional = nn.Parameter(torch.zeros(1, config.max_seq_len // max(config.coarse_factor, 1) + 2, config.model_dim))
        self.coarse_proj = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.dropout = nn.Dropout(config.dropout)
        blocks: list[OmniBlock] = []
        schedule = []
        for i in range(config.depth):
            dilation = config.conv_dilations[i % len(config.conv_dilations)]
            enable_cross = config.cross_every > 0 and (i + 1) % config.cross_every == 0
            schedule.append(enable_cross)
            blocks.append(
                OmniBlock(
                    dim=config.model_dim,
                    heads=config.heads,
                    dropout=config.dropout,
                    ff_mult=config.ff_mult,
                    state_dim=config.ssm_state_dim,
                    ssm_kernel=config.ssm_kernel_size,
                    conv_kernel=config.conv_kernel_size,
                    dilation=dilation,
                    enable_cross=enable_cross,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.cross_schedule = schedule
        self.norm = nn.LayerNorm(config.model_dim)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.coarse_factor <= 1:
            return x
        pooled = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=self.config.coarse_factor,
            stride=self.config.coarse_factor,
            ceil_mode=True,
        )
        return pooled.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {t} exceeds maximum positional encoding {self.config.max_seq_len}"
            )
        fine = self.embed(x)
        pos = self.positional_encoding[:, :t, :]
        fine = self.dropout(fine + pos)

        coarse = self._downsample(fine)
        coarse_len = coarse.size(1)
        coarse_pos = self.coarse_positional[:, :coarse_len, :]
        coarse = self.dropout(self.coarse_proj(coarse + coarse_pos))

        for block, use_cross in zip(self.blocks, self.cross_schedule):
            context = coarse if use_cross else None
            fine = block(fine, context)
            if use_cross:
                coarse = self._downsample(fine)
                coarse_len = coarse.size(1)
                coarse_pos = self.coarse_positional[:, :coarse_len, :]
                coarse = self.dropout(self.coarse_proj(coarse + coarse_pos))

        return self.norm(fine)


def initialise_omni_backbone(feature_dim: int, **overrides: Any) -> MarketOmniBackbone:
    """Utility to instantiate and initialise :class:`MarketOmniBackbone`."""

    config_dict: dict[str, Any] = {"feature_dim": feature_dim}
    config_dict.update(overrides)
    config = OmniBackboneConfig(**config_dict)
    backbone = MarketOmniBackbone(config)
    for name, param in backbone.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    return backbone


__all__ = [
    "OmniBackboneConfig",
    "MarketOmniBackbone",
    "initialise_omni_backbone",
]
