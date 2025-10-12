"""Hybrid transformer / state-space backbone for market modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn


class PatchEmbedding(nn.Module):
    """Project raw features into the model dimension via temporal patches."""

    def __init__(self, input_size: int, d_model: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        if patch_size == 1:
            self.proj = nn.Linear(input_size, d_model)
        else:
            self.proj = nn.Conv1d(input_size, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            return self.proj(x)
        x = rearrange(x, "b t f -> b f t")
        x = self.proj(x)
        return rearrange(x, "b d t -> b t d")


class TemporalConvMixer(nn.Module):
    """Dilated depthwise separable convolution for local pattern capture."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = rearrange(x, "b t d -> b d t")
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = rearrange(x, "b d t -> b t d")
        return residual + self.dropout(x)


class StateSpaceMixer(nn.Module):
    """Light-weight state-space inspired mixer for long horizon memory."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(d_model))
        self.beta = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exponential smoothing per feature dimension implemented in parallel.
        alpha = torch.sigmoid(self.alpha).view(1, 1, -1)
        beta = self.beta.view(1, 1, -1)
        cumulative = torch.cumsum(alpha * x, dim=1)
        filtered = cumulative * beta + x
        return x + self.dropout(filtered)


class GatedResidualNetwork(nn.Module):
    """Gated residual unit used by Temporal Fusion Transformers."""

    def __init__(self, d_model: int, expansion: int, dropout: float):
        super().__init__()
        hidden = d_model * expansion
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(x))
        return residual + h * gate


class TemporalHybridBlock(nn.Module):
    """Single layer mixing attention, convolution, and state-space dynamics."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        conv_kernel: int,
        conv_dilation: int,
        ffn_expansion: int,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.conv_mixer = TemporalConvMixer(d_model, conv_kernel, conv_dilation, dropout)
        self.ssm = StateSpaceMixer(d_model, dropout)
        self.grn = GatedResidualNetwork(d_model, ffn_expansion, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention branch
        attn_input = self.attn_norm(x)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input, attn_mask=attn_mask)
        x = x + self.attn_dropout(attn_out)

        # Convolutional mixer for local context
        x = self.conv_mixer(x)

        # State-space mixer for long memory
        x = self.ssm(x)

        # Gated feed-forward
        return self.grn(x)


@dataclass(slots=True)
class TemporalBackboneConfig:
    """Configuration for the hybrid temporal backbone."""

    input_size: int
    d_model: int = 512
    depth: int = 16
    n_heads: int = 8
    patch_size: int = 1
    dropout: float = 0.1
    conv_kernel: int = 5
    conv_dilations: tuple[int, ...] = (1, 2, 4, 8)
    ffn_expansion: int = 4
    forecast_horizon: int = 5
    output_size: int = 3


class TemporalBackbone(nn.Module):
    """High-capacity backbone for forecasting and trading decisions."""

    def __init__(self, config: TemporalBackboneConfig):
        super().__init__()
        self.config = config
        self.embedding = PatchEmbedding(config.input_size, config.d_model, config.patch_size)
        self.positional = nn.Parameter(torch.randn(1, 4096, config.d_model) * 0.01)

        self.layers = nn.ModuleList()
        dilations = list(config.conv_dilations)
        if len(dilations) < config.depth:
            # Repeat pattern to match depth.
            repeats = (config.depth + len(dilations) - 1) // len(dilations)
            dilations = (dilations * repeats)[: config.depth]

        for dilation in dilations[: config.depth]:
            self.layers.append(
                TemporalHybridBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    dropout=config.dropout,
                    conv_kernel=config.conv_kernel,
                    conv_dilation=dilation,
                    ffn_expansion=config.ffn_expansion,
                )
            )

        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.output_size * config.forecast_horizon),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the backbone."""

        batch, time, _ = x.shape
        h = self.embedding(x)
        if self.positional.size(1) < h.size(1):
            raise ValueError("Sequence length exceeds learned positional encoding capacity.")
        h = h + self.positional[:, : h.size(1)]

        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        h = self.norm(h)
        pooled = h[:, -1]
        logits = self.head(pooled)
        return logits.view(batch, self.config.forecast_horizon, self.config.output_size)
