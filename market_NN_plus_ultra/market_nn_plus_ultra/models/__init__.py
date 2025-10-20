"""Model architectures for Market NN Plus Ultra."""

from .temporal_transformer import TemporalBackbone, TemporalBackboneConfig, TemporalPolicyHead, count_parameters
from .temporal_fusion import TemporalFusionConfig, TemporalFusionTransformer, initialise_temporal_fusion
from .omni_mixture import MarketOmniBackbone, OmniBackboneConfig, initialise_omni_backbone
from .moe_transformer import (
    MixtureOfExpertsBackbone,
    MixtureOfExpertsConfig,
    initialise_moe_backbone,
)
from .state_space import (
    StateSpaceBackbone,
    StateSpaceBlock,
    StateSpaceConfig,
    StateSpaceMixer,
    initialise_state_space_backbone,
)
from .losses import composite_trading_loss, sharpe_ratio_loss
from .calibration import CalibratedPolicyHead, CalibrationHeadOutput
from .market_state import MarketStateEmbedding, MarketStateMetadata, MarketStateFeature

__all__ = [
    "TemporalBackbone",
    "TemporalBackboneConfig",
    "TemporalPolicyHead",
    "count_parameters",
    "TemporalFusionConfig",
    "TemporalFusionTransformer",
    "initialise_temporal_fusion",
    "MarketOmniBackbone",
    "OmniBackboneConfig",
    "initialise_omni_backbone",
    "MixtureOfExpertsBackbone",
    "MixtureOfExpertsConfig",
    "initialise_moe_backbone",
    "StateSpaceBackbone",
    "StateSpaceBlock",
    "StateSpaceMixer",
    "StateSpaceConfig",
    "initialise_state_space_backbone",
    "composite_trading_loss",
    "sharpe_ratio_loss",
    "CalibratedPolicyHead",
    "CalibrationHeadOutput",
    "MarketStateEmbedding",
    "MarketStateMetadata",
    "MarketStateFeature",
]
