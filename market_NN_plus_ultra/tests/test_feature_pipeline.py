import logging

import pandas as pd
import pytest

from market_nn_plus_ultra.data.feature_pipeline import FeatureDependencyError, FeaturePipeline, FeatureSpec
from market_nn_plus_ultra.utils.logging import get_structured_logger


def test_feature_pipeline_records_missing_dependencies() -> None:
    spec = FeatureSpec(
        name="requires_volume",
        function=lambda df: df["volume"],
        depends_on=["volume"],
    )
    pipeline = FeaturePipeline([spec])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    transformed = pipeline.transform(df)

    assert "close" in transformed.columns
    missing = pipeline.get_missing_dependencies()
    assert "requires_volume" in missing
    assert missing["requires_volume"] == ["volume"]


def test_feature_pipeline_strict_mode_raises() -> None:
    spec = FeatureSpec(
        name="needs_high",
        function=lambda df: df["high"],
        depends_on=["high"],
    )
    pipeline = FeaturePipeline([spec], strict=True)
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(FeatureDependencyError):
        pipeline.transform(df)


def test_feature_pipeline_accepts_structured_logger() -> None:
    spec = FeatureSpec(
        name="needs_low",
        function=lambda df: df["low"],
        depends_on=["low"],
    )
    logger = get_structured_logger("test_feature_pipeline", level=logging.DEBUG)
    pipeline = FeaturePipeline([spec], logger=logger)
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    pipeline.transform(df)
    missing = pipeline.get_missing_dependencies()
    assert missing["needs_low"] == ["low"]
