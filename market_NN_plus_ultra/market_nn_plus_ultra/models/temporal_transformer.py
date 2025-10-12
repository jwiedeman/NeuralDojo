"""Deep temporal architecture for market modelling."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


class MultiResolutionAttention(nn.Module):
    """Stacked attention module operating on multiple temporal resolutions."""

    def __init__(self, d_model: int, n_heads: int, dilation_rates: tuple[int, ...]):
        super().__init__()
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in dilation_rates]
        )
        self.dilation_rates = dilation_rates
        self.proj = nn.Linear(len(dilation_rates) * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for rate, attn in zip(self.dilation_rates, self.attentions):
            if rate > 1:
                dilated = x[:, ::rate]
            else:
                dilated = x
            attended, _ = attn(dilated, dilated, dilated)
            if rate > 1:
                expanded = torch.zeros_like(x)
                expanded[:, ::rate] = attended
                outputs.append(expanded)
            else:
                outputs.append(attended)
        concatenated = torch.cat(outputs, dim=-1)
        return self.proj(concatenated)


class GatedResidualUnit(nn.Module):
    """Residual block with gating to stabilise deep stacks."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.act = nn.GELU()
        self.gate = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.norm(x))
        g = torch.sigmoid(self.gate(self.norm(x)))
        x = x + self.fc2(self.act(h) * g)
        return x


@dataclass
class TemporalBackboneConfig:
    input_size: int
    d_model: int = 256
    depth: int = 12
    n_heads: int = 8
    dilation_rates: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    forecast_horizon: int = 1
    output_size: int = 3  # e.g. long/flat/short logits


class TemporalBackbone(nn.Module):
    """Transformer-inspired backbone for market understanding."""

    def __init__(self, config: TemporalBackboneConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_size, config.d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1024, config.d_model) * 0.01)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": MultiResolutionAttention(config.d_model, config.n_heads, config.dilation_rates),
                        "ffn": GatedResidualUnit(config.d_model),
                        "dropout": nn.Dropout(config.dropout),
                    }
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_size * config.forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor shaped (batch, time, features)
        Returns:
            Tensor shaped (batch, horizon, output_size).
        """

        batch, time, _ = x.shape
        pos = self.positional_encoding[:, :time]
        h = self.input_projection(x) + pos
        for layer in self.layers:
            h = h + layer["dropout"](layer["attention"](h))
            h = layer["ffn"](h)
        h = self.norm(h)
        pooled = h[:, -1]  # take last step representation
        logits = self.head(pooled)
        return logits.view(batch, self.config.forecast_horizon, self.config.output_size)
