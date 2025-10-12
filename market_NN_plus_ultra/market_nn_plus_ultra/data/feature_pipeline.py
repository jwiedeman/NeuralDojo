"""Feature engineering pipeline for transforming raw market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


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

    feature_fns: Mapping[str, callable]

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Run all registered feature functions and merge outputs."""

        enriched = panel.copy()
        for name, fn in self.feature_fns.items():
            result = fn(enriched)
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

    @classmethod
    def with_default_indicators(cls) -> "FeaturePipeline":
        """Return a pipeline seeded with common high-signal indicators."""

        def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
            delta = df.groupby(level=0)["close"].diff()
            gain = delta.clip(lower=0).rolling(window).mean()
            loss = -delta.clip(upper=0).rolling(window).mean()
            rs = gain / (loss + 1e-9)
            return 100 - (100 / (1 + rs))

        def rolling_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
            return df.groupby(level=0)["close"].pct_change().rolling(window).std() * np.sqrt(252)

        return cls(
            feature_fns={
                "rsi": rsi,
                "annualized_vol": rolling_vol,
            },
        )
