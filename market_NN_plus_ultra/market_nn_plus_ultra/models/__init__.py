"""Model architectures for Market NN Plus Ultra."""

from .temporal_transformer import TemporalBackbone, TemporalBackboneConfig, TemporalPolicyHead, count_parameters
from .temporal_fusion import TemporalFusionConfig, TemporalFusionTransformer, initialise_temporal_fusion
from .omni_mixture import MarketOmniBackbone, OmniBackboneConfig, initialise_omni_backbone
from .losses import composite_trading_loss, sharpe_ratio_loss

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
    "composite_trading_loss",
    "sharpe_ratio_loss",
]
