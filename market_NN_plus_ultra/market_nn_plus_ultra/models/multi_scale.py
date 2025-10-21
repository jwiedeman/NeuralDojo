"""Hierarchical multi-scale backbone mixing coarse and fine temporal features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .temporal_transformer import TemporalBlock


@dataclass(slots=True)
class MultiScaleBackboneConfig:
    """Configuration for :class:`MultiScaleBackbone`."""

    feature_dim: int
    model_dim: int = 512
    scales: tuple[int, ...] = (1, 4, 16)
    depth_per_scale: int = 4
    heads: int = 8
    dropout: float = 0.1
    conv_kernel_size: int = 5
    conv_dilations: tuple[int, ...] = (1, 2, 4, 8)
    max_seq_len: int = 4096
    fusion_heads: int = 4
    use_rotary_embeddings: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if not self.scales:
            raise ValueError("scales must contain at least one scale factor")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("scale factors must be positive integers")
        if 1 not in self.scales:
            raise ValueError("at least one scale factor must be 1 to preserve base resolution")
        if self.depth_per_scale <= 0:
            raise ValueError("depth_per_scale must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")


class _ScaleEncoder(nn.Module):
    """Stack of :class:`TemporalBlock` layers for a single scale."""

    def __init__(self, config: MultiScaleBackboneConfig, scale: int) -> None:
        super().__init__()
        blocks: list[TemporalBlock] = []
        for layer in range(config.depth_per_scale):
            dilation = config.conv_dilations[layer % len(config.conv_dilations)] * max(scale, 1)
            blocks.append(
                TemporalBlock(
                    dim=config.model_dim,
                    heads=config.heads,
                    dropout=config.dropout,
                    kernel_size=config.conv_kernel_size,
                    dilation=dilation,
                    use_rotary=config.use_rotary_embeddings,
                    max_seq_len=config.max_seq_len,
                    rope_theta=config.rope_theta,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _MultiScaleFusion(nn.Module):
    """Fuse representations across scales with learnable gates and attention."""

    def __init__(self, dim: int, num_scales: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.gates = nn.Parameter(torch.zeros(num_scales))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _upsample_to(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        up = F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return up.transpose(1, 2)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("features must contain at least one tensor")
        base_len = features[0].size(1)
        upsampled = [self._upsample_to(feat, base_len) for feat in features]
        weights = torch.softmax(self.gates, dim=0)
        fused = torch.zeros_like(upsampled[0])
        for weight, feat in zip(weights, upsampled):
            fused = fused + weight * feat
        context = torch.stack(upsampled, dim=0).mean(dim=0)
        attn_out, _ = self.attn(fused, context, context, need_weights=False)
        fused = fused + self.dropout(attn_out)
        fused = self.norm(self.proj(fused))
        return fused


class MultiScaleBackbone(nn.Module):
    """Hierarchical transformer backbone that blends multiple temporal resolutions."""

    def __init__(self, config: MultiScaleBackboneConfig) -> None:
        super().__init__()
        self.config = config
        sorted_scales = sorted(config.scales)
        self.scales = tuple(sorted_scales)
        self.embed = nn.Linear(config.feature_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.positional_embeddings = nn.ParameterList()
        self.encoders = nn.ModuleList()
        for scale in self.scales:
            max_len = (config.max_seq_len + scale - 1) // scale
            self.positional_embeddings.append(nn.Parameter(torch.zeros(1, max_len, config.model_dim)))
            self.encoders.append(_ScaleEncoder(config, scale))
        self.fusion = _MultiScaleFusion(config.model_dim, len(self.scales), config.fusion_heads, config.dropout)
        self.output_norm = nn.LayerNorm(config.model_dim)

    def _downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return x
        pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale, ceil_mode=True)
        return pooled.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum positional encoding {self.config.max_seq_len}"
            )
        embedded = self.embed(x)
        outputs: list[torch.Tensor] = []
        for scale, pos_embed, encoder in zip(self.scales, self.positional_embeddings, self.encoders):
            down = self._downsample(embedded, scale)
            pos = pos_embed[:, : down.size(1), :]
            hidden = self.dropout(down + pos)
            hidden = encoder(hidden)
            outputs.append(hidden)
        fused = self.fusion(outputs)
        return self.output_norm(fused)


def initialise_multi_scale_backbone(feature_dim: int, **overrides: Any) -> MultiScaleBackbone:
    """Helper to instantiate :class:`MultiScaleBackbone` with deterministic initialisation."""

    config_dict = {"feature_dim": feature_dim}
    config_dict.update(overrides)
    config = MultiScaleBackboneConfig(**config_dict)
    backbone = MultiScaleBackbone(config)
    for name, param in backbone.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    return backbone


__all__ = [
    "MultiScaleBackboneConfig",
    "MultiScaleBackbone",
    "initialise_multi_scale_backbone",
]
