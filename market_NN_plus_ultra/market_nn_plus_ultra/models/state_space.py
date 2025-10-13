"""State-space backbone for Market NN Plus Ultra.

This module provides a lightweight yet expressive approximation of structured
state-space models (SSM) tailored for long-context financial sequences.  The
architecture follows a residual stack that alternates gated depthwise
convolutions with feed-forward expansions, inspired by modern SSM literature
(S4/S5, DSS) while keeping the implementation dependency free.

The design intentionally mirrors the configuration knobs used by the other
backbones so experiments can swap architectures without modifying training
loops.  The mixer uses a learned gating mechanism to interpolate between the
previous state and the new convolutional response, providing stable gradients
for extremely deep stacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn


@dataclass(slots=True)
class StateSpaceConfig:
    """Configuration for :class:`StateSpaceBackbone`."""

    feature_dim: int
    model_dim: int = 512
    depth: int = 12
    state_dim: int = 256
    kernel_size: int = 15
    dropout: float = 0.1
    ff_mult: int = 4
    max_seq_len: int = 4096


class StateSpaceMixer(nn.Module):
    """Depthwise convolutional mixer with learnable gating."""

    def __init__(self, dim: int, state_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size - 1
        self.input_proj = nn.Linear(dim, state_dim * 2)
        self.conv = nn.Conv1d(
            state_dim,
            state_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=state_dim,
        )
        self.output_proj = nn.Linear(state_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, dim]
        gate, state = self.input_proj(x).chunk(2, dim=-1)
        state = rearrange(state, "b t d -> b d t")
        state = self.conv(state)
        # remove the look-back padding so the output length matches the input
        state = state[..., : x.size(1)]
        state = rearrange(state, "b d t -> b t d")
        state = torch.tanh(state)
        gate = torch.sigmoid(gate)
        mixed = gate * state
        mixed = self.output_proj(mixed)
        return self.dropout(mixed)


class StateSpaceBlock(nn.Module):
    """Residual block combining an SSM-inspired mixer and feed-forward network."""

    def __init__(self, dim: int, state_dim: int, kernel_size: int, dropout: float, ff_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = StateSpaceMixer(dim, state_dim, kernel_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class StateSpaceBackbone(nn.Module):
    """Backbone that stacks :class:`StateSpaceBlock` layers."""

    def __init__(self, config: StateSpaceConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Linear(config.feature_dim, config.model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.model_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                StateSpaceBlock(
                    dim=config.model_dim,
                    state_dim=config.state_dim,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                    ff_mult=config.ff_mult,
                )
                for _ in range(config.depth)
            ]
        )
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
        x = self.dropout(x + pos)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def initialise_state_space_backbone(feature_dim: int, **overrides) -> StateSpaceBackbone:
    config_dict = {"feature_dim": feature_dim}
    config_dict.update(overrides)
    config = StateSpaceConfig(**config_dict)
    backbone = StateSpaceBackbone(config)
    for name, param in backbone.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    return backbone


__all__ = [
    "StateSpaceConfig",
    "StateSpaceBackbone",
    "StateSpaceBlock",
    "StateSpaceMixer",
    "initialise_state_space_backbone",
]
