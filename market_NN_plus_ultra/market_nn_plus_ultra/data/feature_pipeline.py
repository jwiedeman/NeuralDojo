"""Feature engineering pipeline for transforming raw market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .feature_registry import FeatureFn, FeatureSpec


@dataclass(slots=True)
class FeaturePipeline:
    """Applies technical indicators and advanced signals to price panels.

    The pipeline keeps a registry of feature functions. Each function receives a
    pandas ``DataFrame`` and returns either a ``Series`` or a ``DataFrame`` that
    will be joined back to the panel. Implementations can include:

    * Classical indicators (RSI, MACD, Bollinger bands).
    * Spectral/transformation features (wavelets, FFT magnitude bands).
    * Learned embeddings (news sentiment, chain-of-thought policy hints).
    * Regime and volatility detectors.
    """

    feature_fns: Mapping[str, FeatureFn | FeatureSpec]

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Run all registered feature functions and merge outputs."""

        enriched = panel.copy()
        for name, entry in self.feature_fns.items():
            spec = self._resolve(entry)
            missing = [dep for dep in spec.depends_on if dep not in enriched.columns]
            if missing:
                # Skip gracefully when dependencies are not satisfied.
                continue
            result = spec.fn(enriched)
            if isinstance(result, pd.Series):
                enriched[name] = result
            elif isinstance(result, pd.DataFrame):
                for column in result.columns:
                    enriched[f"{name}_{column}"] = result[column]
            else:
                raise TypeError(f"Feature '{name}' returned unsupported type {type(result)!r}")
        enriched.replace([np.inf, -np.inf], np.nan, inplace=True)
        enriched.sort_index(inplace=True)
        return enriched

    @staticmethod
    def _resolve(entry: FeatureFn | FeatureSpec) -> FeatureSpec:
        if isinstance(entry, FeatureSpec):
            return entry
        return FeatureSpec(fn=entry)

    def describe(self) -> dict[str, dict[str, Iterable[str]]]:
        """Expose metadata about the underlying features."""

        description: dict[str, dict[str, Iterable[str]]] = {}
        for name, entry in self.feature_fns.items():
            spec = self._resolve(entry)
            description[name] = {
                "tags": tuple(spec.tags),
                "depends_on": tuple(spec.depends_on),
            }
        return description

    @classmethod
    def with_default_indicators(cls) -> "FeaturePipeline":
        """Return a pipeline seeded with common high-signal indicators."""

        from .feature_registry import FeatureRegistry

        registry = FeatureRegistry.default()
        return registry.build_pipeline()
