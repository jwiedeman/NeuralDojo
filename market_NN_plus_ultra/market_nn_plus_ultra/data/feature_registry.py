"""Registry for engineered features and indicator functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import pandas as pd

FeatureFn = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Metadata wrapper for a single engineered feature.

    Attributes:
        fn: Callable that receives the full multi-indexed panel and returns a
            ``Series`` or ``DataFrame`` aligned with the original index.
        description: Short human readable description for documentation / UIs.
        tags: Optional tuple of keywords describing the feature family
            (e.g. ``("momentum", "volatility")``).
        depends_on: Optional tuple of column names that must be present in the
            panel prior to feature computation. Missing dependencies cause the
            feature to be skipped gracefully.
    """

    fn: FeatureFn
    description: str = ""
    tags: Tuple[str, ...] = ()
    depends_on: Tuple[str, ...] = ()


class FeatureRegistry:
    """Mutable registry mapping feature names to :class:`FeatureSpec` objects."""

    def __init__(self, specs: Optional[Mapping[str, FeatureSpec]] = None):
        self._specs: MutableMapping[str, FeatureSpec] = dict(specs or {})

    # ------------------------------------------------------------------
    # mutation helpers
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        fn: FeatureFn,
        *,
        description: str = "",
        tags: Iterable[str] | None = None,
        depends_on: Iterable[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a new feature in the registry."""

        if not overwrite and name in self._specs:
            raise ValueError(f"Feature '{name}' already registered")
        self._specs[name] = FeatureSpec(
            fn=fn,
            description=description,
            tags=tuple(tags or ()),
            depends_on=tuple(depends_on or ()),
        )

    def update(self, mapping: Mapping[str, FeatureSpec]) -> None:
        for name, spec in mapping.items():
            self._specs[name] = spec

    # ------------------------------------------------------------------
    # retrieval helpers
    # ------------------------------------------------------------------
    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def names(self) -> Tuple[str, ...]:
        return tuple(self._specs.keys())

    def describe(self) -> Dict[str, dict]:
        """Return JSON-serialisable metadata for dashboards or docs."""

        return {
            name: {
                "description": spec.description,
                "tags": list(spec.tags),
                "depends_on": list(spec.depends_on),
            }
            for name, spec in self._specs.items()
        }

    # ------------------------------------------------------------------
    # pipeline integration
    # ------------------------------------------------------------------
    def build_pipeline(self, selected: Optional[Iterable[str]] = None):
        """Return a :class:`FeaturePipeline` configured with registry entries."""

        from .feature_pipeline import FeaturePipeline  # lazy import to avoid cycle

        if selected is None:
            selected_names = list(self._specs.keys())
        else:
            selected_names = [name for name in selected if name in self._specs]
        feature_map = {name: self._specs[name] for name in selected_names}
        return FeaturePipeline(feature_fns=feature_map)

    # ------------------------------------------------------------------
    # defaults
    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "FeatureRegistry":
        """Return a registry seeded with research-grade indicators."""

        registry = cls()

        def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
            prices = df["close"]
            delta = prices.groupby(level=0).diff()
            gain = delta.clip(lower=0).groupby(level=0).ewm(alpha=1 / window, adjust=False).mean()
            loss = -delta.clip(upper=0).groupby(level=0).ewm(alpha=1 / window, adjust=False).mean()
            rs = gain / (loss + 1e-9)
            return 100 - (100 / (1 + rs))

        registry.register(
            "rsi",
            rsi,
            description="Relative Strength Index with exponential smoothing",
            tags=("momentum",),
            depends_on=("close",),
        )

        def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
            close = df["close"]
            ema_fast = close.groupby(level=0).ewm(span=fast, adjust=False).mean()
            ema_slow = close.groupby(level=0).ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.groupby(level=0).ewm(span=signal, adjust=False).mean()
            hist = macd_line - signal_line
            return pd.DataFrame({
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "macd_hist": hist,
            })

        registry.register(
            "macd",
            macd,
            description="Moving Average Convergence Divergence (EMA 12/26/9)",
            tags=("trend", "momentum"),
            depends_on=("close",),
        )

        def bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
            close = df["close"]
            rolling = close.groupby(level=0).rolling(window=window, min_periods=window)
            mean = rolling.mean()
            std = rolling.std()
            upper = mean + n_std * std
            lower = mean - n_std * std
            width = (upper - lower) / (mean + 1e-9)
            return pd.DataFrame({
                "bb_upper": upper,
                "bb_lower": lower,
                "bb_width": width,
            })

        registry.register(
            "bollinger",
            bollinger,
            description="Bollinger bands with dynamic width",
            tags=("volatility", "trend"),
            depends_on=("close",),
        )

        def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            prev_close = close.groupby(level=0).shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.groupby(level=0).ewm(span=window, adjust=False).mean()
            return atr_series

        registry.register(
            "atr",
            atr,
            description="Average True Range using Wilder smoothing",
            tags=("volatility", "risk"),
            depends_on=("high", "low", "close"),
        )

        def volume_zscore(df: pd.DataFrame, window: int = 60) -> pd.Series:
            volume = df["volume"]
            rolling = volume.groupby(level=0).rolling(window=window, min_periods=window)
            mean = rolling.mean()
            std = rolling.std().replace(0, 1.0)
            z = (volume - mean) / (std + 1e-9)
            return z

        registry.register(
            "volume_zscore",
            volume_zscore,
            description="Normalised volume anomaly score",
            tags=("volume", "regime"),
            depends_on=("volume",),
        )

        def realized_vol(df: pd.DataFrame, window: int = 30) -> pd.Series:
            returns = df.groupby(level=0)["close"].pct_change()
            vol = returns.groupby(level=0).rolling(window=window, min_periods=window).std()
            return vol * (252 ** 0.5)

        registry.register(
            "realized_vol",
            realized_vol,
            description="Annualised realised volatility of close-to-close returns",
            tags=("volatility", "risk"),
            depends_on=("close",),
        )

        def regime_probabilities(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
            returns = df.groupby(level=0)["close"].pct_change().fillna(0.0)
            mu = returns.groupby(level=0).rolling(window=window, min_periods=window).mean()
            sigma = (
                returns.groupby(level=0)
                .rolling(window=window, min_periods=window)
                .std()
                .replace(0, 1e-6)
            )
            z = (returns - mu) / sigma
            bull = (z > 0.5).astype(float)
            bear = (z < -0.5).astype(float)
            neutral = 1.0 - bull - bear
            return pd.DataFrame({
                "regime_bull": bull,
                "regime_bear": bear,
                "regime_neutral": neutral,
            })

        registry.register(
            "regime",
            regime_probabilities,
            description="Soft regime classification based on return z-scores",
            tags=("regime", "risk"),
            depends_on=("close",),
        )

        return registry


__all__ = ["FeatureRegistry", "FeatureSpec", "FeatureFn"]
