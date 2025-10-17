"""Data ingestion and preprocessing utilities for Market NN Plus Ultra."""

from .sqlite_loader import SQLiteMarketSource, SQLiteMarketDataset
from .alternative_data import AlternativeDataConnector, AlternativeDataSpec
from .feature_pipeline import FeatureDependencyError, FeatureRegistry, FeaturePipeline, FeatureSpec
from .window_dataset import SlidingWindowDataset
from .validation import validate_price_frame, validate_indicator_frame
from .fixtures import FixtureConfig, build_fixture, write_fixture

__all__ = [
    "SQLiteMarketSource",
    "SQLiteMarketDataset",
    "FeatureRegistry",
    "FeaturePipeline",
    "FeatureSpec",
    "AlternativeDataSpec",
    "AlternativeDataConnector",
    "FeatureDependencyError",
    "SlidingWindowDataset",
    "validate_price_frame",
    "validate_indicator_frame",
    "FixtureConfig",
    "build_fixture",
    "write_fixture",
]
