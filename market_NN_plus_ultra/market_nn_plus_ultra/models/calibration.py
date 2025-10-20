"""Calibration-aware policy heads for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _erfinv(x: Tensor) -> Tensor:
    """Stable inverse error function wrapper supporting `float32` tensors."""

    # PyTorch exposes `torch.special.erfinv` starting from 1.8; mirror the call here so
    # downstream callers do not need to remember the namespace.
    return torch.special.erfinv(x)


@dataclass(slots=True)
class CalibrationHeadOutput:
    """Outputs returned by :class:`CalibratedPolicyHead`."""

    prediction: Tensor
    quantiles: Tensor
    quantile_levels: Tensor
    concentration: Tensor

    def probabilities(self) -> Tensor:
        """Return normalised Dirichlet probabilities for each horizon step."""

        total = self.concentration.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.concentration / total

    def expected(self) -> Tensor:
        """Alias for the mean prediction."""

        return self.prediction


class CalibratedPolicyHead(nn.Module):
    """Projection head that emits calibrated forecasts and uncertainty."""

    def __init__(
        self,
        model_dim: int,
        horizon: int,
        output_dim: int,
        *,
        quantile_levels: Sequence[float] = (0.05, 0.5, 0.95),
        dirichlet_temperature: float = 1.0,
        min_concentration: float = 1e-2,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive for CalibratedPolicyHead")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive for CalibratedPolicyHead")
        if dirichlet_temperature <= 0:
            raise ValueError("dirichlet_temperature must be positive")
        if min_concentration <= 0:
            raise ValueError("min_concentration must be positive")

        if not quantile_levels:
            raise ValueError("quantile_levels must contain at least one entry")
        levels = torch.tensor(list(quantile_levels), dtype=torch.float32)
        if levels.ndim != 1:
            raise ValueError("quantile_levels must be a 1D sequence")
        if torch.any((levels <= 0) | (levels >= 1)):
            raise ValueError("quantile_levels must be strictly between 0 and 1")
        sorted_levels, _ = torch.sort(levels)
        self.register_buffer("quantile_levels", sorted_levels, persistent=False)

        # Pre-compute the standard normal inverse CDF for each quantile so the head can
        # emit monotonic quantile predictions by scaling a single positive width term.
        standard_basis = torch.sqrt(torch.tensor(2.0, dtype=torch.float32)) * _erfinv(
            2 * sorted_levels - 1
        )
        self.register_buffer("_quantile_basis", standard_basis, persistent=False)

        self.horizon = horizon
        self.output_dim = output_dim
        self.dirichlet_temperature = float(dirichlet_temperature)
        self.min_concentration = float(min_concentration)

        self.pre = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
        )
        self.mean_proj = nn.Linear(model_dim, horizon * output_dim)
        self.scale_proj = nn.Linear(model_dim, horizon * output_dim)
        self.concentration_proj = nn.Linear(model_dim, horizon * output_dim)

        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> CalibrationHeadOutput:
        # x: [batch, time, dim] – we pool the final timestep for decisions.
        last_state = x[:, -1, :]
        hidden = self.pre(last_state)
        batch = hidden.size(0)

        mean = self.mean_proj(hidden).view(batch, self.horizon, self.output_dim)
        raw_scale = self.scale_proj(hidden).view(batch, self.horizon, self.output_dim)
        scale = F.softplus(raw_scale) + 1e-3

        quantile_offsets = scale.unsqueeze(-1) * self._quantile_basis.view(1, 1, 1, -1)
        quantiles = mean.unsqueeze(-1) + quantile_offsets

        raw_concentration = self.concentration_proj(hidden).view(
            batch, self.horizon, self.output_dim
        )
        concentration = (
            F.softplus(raw_concentration / self.dirichlet_temperature)
            + self.min_concentration
        )

        return CalibrationHeadOutput(
            prediction=mean,
            quantiles=quantiles,
            quantile_levels=self.quantile_levels,
            concentration=concentration,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def iter_parameters(self) -> Iterable[nn.Parameter]:
        """Return an iterator over learnable parameters (mostly for tests)."""

        return self.parameters()


__all__ = ["CalibratedPolicyHead", "CalibrationHeadOutput"]

