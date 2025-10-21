"""Statistical drift diagnostics for live monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class DriftMetrics:
    """Container holding common drift statistics."""

    population_stability_index: float
    js_divergence: float
    ks_statistic: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a serialisable dictionary."""

        return {
            "population_stability_index": float(self.population_stability_index),
            "js_divergence": float(self.js_divergence),
            "ks_statistic": float(self.ks_statistic),
        }


def _to_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def _histogram_probabilities(samples: np.ndarray, bins: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if samples.size == 0:
        return np.asarray([0.0]), np.asarray([1.0])

    sample_min = float(samples.min())
    sample_max = float(samples.max())
    if np.isclose(sample_min, sample_max):
        sample_max = sample_min + 1.0

    bin_edges = np.linspace(sample_min, sample_max, bins + 1, dtype=np.float64)
    counts, edges = np.histogram(samples, bins=bin_edges)
    probabilities = counts.astype(np.float64) + eps
    probabilities /= probabilities.sum()
    return edges, probabilities


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    ratio = np.where(p > 0, p / (q + eps), 1.0)
    return float(np.sum(p * np.log(ratio + eps)))


def compute_drift_metrics(
    reference: Iterable[float],
    observed: Iterable[float],
    *,
    bins: int = 20,
    eps: float = 1e-6,
) -> DriftMetrics:
    """Compute drift statistics comparing *observed* to *reference* values."""

    reference_array = _to_array(reference)
    observed_array = _to_array(observed)

    if reference_array.size == 0 or observed_array.size == 0:
        return DriftMetrics(0.0, 0.0, 0.0)

    _, ref_probs = _histogram_probabilities(reference_array, bins, eps)
    _, obs_probs = _histogram_probabilities(observed_array, bins, eps)

    # Align probability vectors by padding to the longest length.
    max_len = max(ref_probs.size, obs_probs.size)
    if ref_probs.size != max_len:
        ref_probs = np.pad(ref_probs, (0, max_len - ref_probs.size), constant_values=eps)
        ref_probs /= ref_probs.sum()
    if obs_probs.size != max_len:
        obs_probs = np.pad(obs_probs, (0, max_len - obs_probs.size), constant_values=eps)
        obs_probs /= obs_probs.sum()

    psi_components = (ref_probs - obs_probs) * np.log((ref_probs + eps) / (obs_probs + eps))
    psi = float(np.sum(psi_components))

    midpoint = 0.5 * (ref_probs + obs_probs)
    js = 0.5 * (_kl_divergence(ref_probs, midpoint, eps) + _kl_divergence(obs_probs, midpoint, eps))

    # Kolmogorov–Smirnov statistic based on empirical CDF differences.
    sorted_ref = np.sort(reference_array)
    sorted_obs = np.sort(observed_array)
    ref_cdf = np.linspace(1.0 / sorted_ref.size, 1.0, sorted_ref.size)
    obs_cdf = np.linspace(1.0 / sorted_obs.size, 1.0, sorted_obs.size)
    ks = float(np.max(np.abs(np.interp(sorted_ref, sorted_obs, obs_cdf, left=0.0, right=1.0) - ref_cdf)))

    return DriftMetrics(psi, js, ks)


__all__ = ["DriftMetrics", "compute_drift_metrics"]
