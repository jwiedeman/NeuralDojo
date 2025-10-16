"""Hybrid temporal transformer for market modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
    max_seq_len: int = 4096
    use_rotary_embeddings: bool = True
    rope_theta: float = 10000.0


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding cache supporting dynamic sequence lengths."""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even")
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        max_seq = max(seq_len, self.max_seq_len)
        t = torch.arange(max_seq, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self.cos_cached = cos[None, None, :, :]
        self.sin_cached = sin[None, None, :, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = max(q.size(-2), k.size(-2))
        if self.cos_cached.numel() == 0 or self.cos_cached.size(-2) < seq_len:
            self._build_cache(seq_len, q.device, q.dtype)
        cos = self.cos_cached[..., :seq_len, :]
        sin = self.sin_cached[..., :seq_len, :]
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class TemporalAttention(nn.Module):
    """Multi-head self-attention with optional rotary embeddings."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        *,
        use_rotary: bool,
        max_seq_len: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_rotary = use_rotary
        self.rotary: Optional[RotaryEmbedding]
        if use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_theta)
        else:
            self.rotary = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.heads)
        if self.rotary is not None:
            q, k = self.rotary(q, k)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out(out)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        kernel_size: int,
        dilation: int,
        *,
        use_rotary: bool,
        max_seq_len: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.attn = TemporalAttention(
            dim,
            heads,
            dropout,
            use_rotary=use_rotary,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
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
        attn_out = self.attn(x)
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
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.model_dim))
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
                    use_rotary=config.use_rotary_embeddings,
                    max_seq_len=config.max_seq_len,
                    rope_theta=config.rope_theta,
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

