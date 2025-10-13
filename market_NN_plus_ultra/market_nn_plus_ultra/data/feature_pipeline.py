"""Feature engineering pipeline for Market NN Plus Ultra."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.trend import CCIIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

FeatureFunction = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@dataclass(slots=True)
class FeatureSpec:
    """Metadata for a single engineered feature."""

    name: str
    function: FeatureFunction
    depends_on: Iterable[str]
    description: str = ""
    tags: Optional[Iterable[str]] = None


class FeatureRegistry:
    """Registry of engineered market features."""

    def __init__(self) -> None:
        self._registry: Dict[str, FeatureSpec] = {}
        self._bootstrap_defaults()

    def _bootstrap_defaults(self) -> None:
        self.register(
            FeatureSpec(
                name="rsi_14",
                function=lambda df: RSIIndicator(close=df["close"], window=14).rsi(),
                depends_on=["close"],
                description="Relative Strength Index over 14 periods",
                tags=["momentum"],
            )
        )
        self.register(
            FeatureSpec(
                name="macd_hist",
                function=lambda df: MACD(close=df["close"]).macd_diff(),
                depends_on=["close"],
                description="MACD histogram capturing trend accelerations",
                tags=["trend"],
            )
        )
        self.register(
            FeatureSpec(
                name="bollinger_band_width",
                function=lambda df: BollingerBands(close=df["close"], window=20, window_dev=2.0).bollinger_wband(),
                depends_on=["close"],
                description="Relative Bollinger band width as a volatility proxy",
                tags=["volatility"],
            )
        )
        self.register(
            FeatureSpec(
                name="ema_ratio_12_26",
                function=lambda df: self._ema_ratio(df["close"], fast=12, slow=26),
                depends_on=["close"],
                description="Ratio between fast and slow exponential moving averages",
                tags=["trend", "momentum"],
            )
        )
        self.register(
            FeatureSpec(
                name="ema_distance_50",
                function=lambda df: self._ema_distance(df["close"], window=50),
                depends_on=["close"],
                description="Normalised distance from the 50-period EMA",
                tags=["mean_reversion", "momentum"],
            )
        )
        self.register(
            FeatureSpec(
                name="vwap_ratio",
                function=lambda df: df["close"] / VolumeWeightedAveragePrice(
                    high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20
                ).volume_weighted_average_price(),
                depends_on=["high", "low", "close", "volume"],
                description="Price distance from VWAP",
                tags=["volume", "intraday"],
            )
        )
        self.register(
            FeatureSpec(
                name="stochastic_oscillator",
                function=lambda df: self._stochastic_oscillator(df, window=14, smooth_window=3),
                depends_on=["high", "low", "close"],
                description="Stochastic oscillator %K/%D levels",
                tags=["momentum", "bounded"],
            )
        )
        self.register(
            FeatureSpec(
                name="chaikin_money_flow_20",
                function=lambda df: self._chaikin_money_flow(df, window=20),
                depends_on=["high", "low", "close", "volume"],
                description="Chaikin money flow over a 20-period window",
                tags=["volume", "breadth"],
            )
        )
        self.register(
            FeatureSpec(
                name="on_balance_volume",
                function=lambda df: self._on_balance_volume(df["close"], df["volume"]),
                depends_on=["close", "volume"],
                description="Cumulative On-Balance Volume series",
                tags=["volume"],
            )
        )
        self.register(
            FeatureSpec(
                name="true_strength_index",
                function=lambda df: self._true_strength_index(df["close"], window_slow=25, window_fast=13),
                depends_on=["close"],
                description="True Strength Index for momentum confirmation",
                tags=["momentum", "trend"],
            )
        )
        self.register(
            FeatureSpec(
                name="commodity_channel_index_20",
                function=lambda df: self._commodity_channel_index(df, window=20),
                depends_on=["high", "low", "close"],
                description="Commodity Channel Index highlighting cyclical extremes",
                tags=["momentum", "mean_reversion"],
            )
        )
        self.register(
            FeatureSpec(
                name="log_return_1",
                function=lambda df: np.log(df["close"]).diff(1),
                depends_on=["close"],
                description="One-step log return",
                tags=["returns"],
            )
        )
        self.register(
            FeatureSpec(
                name="log_return_5",
                function=lambda df: np.log(df["close"]).diff(5),
                depends_on=["close"],
                description="Five-step log return",
                tags=["returns", "multi_horizon"],
            )
        )
        self.register(
            FeatureSpec(
                name="realised_vol_20",
                function=lambda df: np.log(df["close"]).diff().rolling(20).std() * math.sqrt(252),
                depends_on=["close"],
                description="Annualised realised volatility over 20 periods",
                tags=["volatility"],
            )
        )
        self.register(
            FeatureSpec(
                name="volume_zscore",
                function=lambda df: (df["volume"] - df["volume"].rolling(60).mean()) / df["volume"].rolling(60).std(),
                depends_on=["volume"],
                description="Rolling z-score of volume anomalies",
                tags=["volume", "regime"],
            )
        )
        self.register(
            FeatureSpec(
                name="regime_score",
                function=lambda df: (df["close"] - df["close"].rolling(100).mean()) / df["close"].rolling(100).std(),
                depends_on=["close"],
                description="Soft bull/bear regime score",
                tags=["regime"],
            )
        )
        self.register(
            FeatureSpec(
                name="rolling_skew_30",
                function=lambda df: df["close"].pct_change().rolling(30).skew(),
                depends_on=["close"],
                description="30-step skewness of returns",
                tags=["higher_moment"],
            )
        )
        self.register(
            FeatureSpec(
                name="rolling_kurtosis_30",
                function=lambda df: df["close"].pct_change().rolling(30).kurt(),
                depends_on=["close"],
                description="30-step kurtosis of returns",
                tags=["higher_moment"],
            )
        )
        self.register(
            FeatureSpec(
                name="fft_energy_ratio",
                function=lambda df: self._fft_energy_ratio(df["close"], window=128, top_k=5),
                depends_on=["close"],
                description="Ratio of high-frequency FFT energy to total energy over 128 steps",
                tags=["spectral"],
            )
        )

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._registry:
            raise ValueError(f"Feature '{spec.name}' already registered")
        self._registry[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._registry.pop(name, None)

    def describe(self) -> pd.DataFrame:
        records = []
        for spec in self._registry.values():
            records.append(
                {
                    "name": spec.name,
                    "depends_on": list(spec.depends_on),
                    "description": spec.description,
                    "tags": list(spec.tags or []),
                }
            )
        return pd.DataFrame.from_records(records)

    def build_pipeline(self, selected: Optional[Iterable[str]] = None) -> "FeaturePipeline":
        if selected is None:
            specs = list(self._registry.values())
        else:
            specs = [self._registry[name] for name in selected]
        return FeaturePipeline(specs)

    def to_markdown(
        self,
        path: str | Path,
        *,
        title: str = "Market NN Plus Ultra Feature Registry",
        include_timestamp: bool = True,
    ) -> Path:
        """Export the registry metadata to a Markdown table.

        Parameters
        ----------
        path:
            Target file path. Parent directories are created automatically.
        title:
            Optional heading to place at the top of the document.
        include_timestamp:
            Whether to include the generation timestamp in the document
            preamble for auditability.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.describe().sort_values("name")
        lines: list[str] = []
        if title:
            lines.append(f"# {title}")
            lines.append("")
        if include_timestamp:
            lines.append("_Auto-generated from the live registry. Update by running ``python scripts/export_features.py``._")
            lines.append("")

        headers = ["Name", "Depends On", "Tags", "Description"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        if df.empty:
            lines.append("| _(none)_ | - | - | Registry is currently empty. |")
        else:
            for _, row in df.iterrows():
                depends_on = ", ".join(row["depends_on"]) if row["depends_on"] else "-"
                tags = ", ".join(row["tags"]) if row["tags"] else "-"
                description = row.get("description", "") or "-"
                lines.append(f"| {row['name']} | {depends_on} | {tags} | {description} |")

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _fft_energy_ratio(close: pd.Series, window: int, top_k: int) -> pd.Series:
        def _energy(segment: np.ndarray) -> float:
            spectrum = np.abs(np.fft.rfft(segment - segment.mean()))
            total = np.sum(spectrum)
            if total == 0:
                return 0.0
            top = np.sum(np.sort(spectrum)[-top_k:])
            return float(top / total)

        return close.rolling(window).apply(lambda x: _energy(np.asarray(x)), raw=False)


class FeaturePipeline:
    """Execute a list of feature engineering steps on market panels."""

    def __init__(self, features: Iterable[FeatureSpec]) -> None:
        self.features: List[FeatureSpec] = list(features)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with engineered features appended."""

        feature_columns: MutableMapping[str, pd.Series | pd.DataFrame] = {}
        for spec in self.features:
            missing = [col for col in spec.depends_on if col not in df.columns]
            if missing:
                continue
            values = spec.function(df)
            if isinstance(values, pd.DataFrame):
                for column in values.columns:
                    feature_columns[f"{spec.name}__{column}"] = values[column]
            else:
                feature_columns[spec.name] = values
        if not feature_columns:
            return df
        feature_df = pd.concat(feature_columns, axis=1)
        if isinstance(feature_df.columns, pd.MultiIndex):
            feature_df.columns = [
                "__".join([str(level) for level in col if str(level)])
                for col in feature_df.columns.to_list()
            ]
        return pd.concat([df, feature_df], axis=1)

    def transform_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply the feature pipeline symbol by symbol to a multi-indexed panel."""

        symbols = panel.index.get_level_values("symbol").unique()
        enriched_frames: List[pd.DataFrame] = []
        for symbol in symbols:
            df = panel.xs(symbol, level="symbol").copy()
            enriched = self.transform(df)
            enriched["symbol"] = symbol
            enriched_frames.append(enriched)
        enriched_panel = pd.concat(enriched_frames)
        return enriched_panel.reset_index().set_index(["timestamp", "symbol"]).sort_index()

    @staticmethod
    def _ema_ratio(close: pd.Series, *, fast: int, slow: int) -> pd.Series:
        fast_ema = EMAIndicator(close=close, window=fast).ema_indicator()
        slow_ema = EMAIndicator(close=close, window=slow).ema_indicator()
        return fast_ema / (slow_ema + 1e-9)

    @staticmethod
    def _ema_distance(close: pd.Series, *, window: int) -> pd.Series:
        ema = EMAIndicator(close=close, window=window).ema_indicator()
        return (close - ema) / (ema + 1e-9)

    @staticmethod
    def _stochastic_oscillator(
        df: pd.DataFrame,
        *,
        window: int,
        smooth_window: int,
    ) -> pd.DataFrame:
        oscillator = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=window,
            smooth_window=smooth_window,
        )
        return pd.DataFrame(
            {
                "stoch_k": oscillator.stoch(),
                "stoch_d": oscillator.stoch_signal(),
            }
        )

    @staticmethod
    def _chaikin_money_flow(df: pd.DataFrame, *, window: int) -> pd.Series:
        indicator = ChaikinMoneyFlowIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            window=window,
        )
        return indicator.chaikin_money_flow()

    @staticmethod
    def _on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        indicator = OnBalanceVolumeIndicator(close=close, volume=volume)
        return indicator.on_balance_volume()

    @staticmethod
    def _true_strength_index(close: pd.Series, *, window_slow: int, window_fast: int) -> pd.Series:
        indicator = TSIIndicator(close=close, window_slow=window_slow, window_fast=window_fast)
        return indicator.tsi()

    @staticmethod
    def _commodity_channel_index(df: pd.DataFrame, *, window: int) -> pd.Series:
        indicator = CCIIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=window,
        )
        return indicator.cci()

