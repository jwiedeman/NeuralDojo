"""Data ingestion and preprocessing utilities for Market NN Plus Ultra."""

from .sqlite_loader import SQLiteMarketSource, SQLiteMarketDataset
from .alternative_data import AlternativeDataConnector, AlternativeDataSpec
from .feature_pipeline import FeatureDependencyError, FeatureRegistry, FeaturePipeline, FeatureSpec
from .labelling import (
    LiquidityRegimeConfig,
    MarketRegimeLabellingConfig,
    RotationRegimeConfig,
    VolatilityRegimeConfig,
    generate_regime_labels,
)
from .window_dataset import SlidingWindowDataset
from .validation import (
    validate_assets_frame,
    validate_benchmark_frame,
    validate_indicator_frame,
    validate_price_frame,
    validate_regime_frame,
    validate_sqlite_frames,
    validate_trades_frame,
)
from .fixtures import FixtureConfig, build_fixture, write_fixture

__all__ = [
    "SQLiteMarketSource",
    "SQLiteMarketDataset",
    "FeatureRegistry",
    "FeaturePipeline",
    "FeatureSpec",
    "MarketRegimeLabellingConfig",
    "AlternativeDataSpec",
    "AlternativeDataConnector",
    "FeatureDependencyError",
    "VolatilityRegimeConfig",
    "LiquidityRegimeConfig",
    "RotationRegimeConfig",
    "generate_regime_labels",
    "SlidingWindowDataset",
    "validate_assets_frame",
    "validate_benchmark_frame",
    "validate_indicator_frame",
    "validate_price_frame",
    "validate_regime_frame",
    "validate_sqlite_frames",
    "validate_trades_frame",
    "FixtureConfig",
    "build_fixture",
    "write_fixture",
]
