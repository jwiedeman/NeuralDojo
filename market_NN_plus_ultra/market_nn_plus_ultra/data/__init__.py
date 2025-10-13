"""Data ingestion and preprocessing utilities for Market NN Plus Ultra."""

from .sqlite_loader import SQLiteMarketSource, SQLiteMarketDataset
from .feature_pipeline import FeatureDependencyError, FeatureRegistry, FeaturePipeline, FeatureSpec
from .window_dataset import SlidingWindowDataset
from .validation import validate_price_frame, validate_indicator_frame

__all__ = [
    "SQLiteMarketSource",
    "SQLiteMarketDataset",
    "FeatureRegistry",
    "FeaturePipeline",
    "FeatureSpec",
    "FeatureDependencyError",
    "SlidingWindowDataset",
    "validate_price_frame",
    "validate_indicator_frame",
]
