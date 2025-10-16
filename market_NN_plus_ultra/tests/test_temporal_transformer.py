import torch

from market_nn_plus_ultra.models.temporal_transformer import (
    TemporalBackbone,
    TemporalBackboneConfig,
)


def test_temporal_backbone_forward_rotary():
    config = TemporalBackboneConfig(
        feature_dim=8,
        model_dim=32,
        depth=2,
        heads=4,
        dropout=0.0,
        conv_kernel_size=3,
        conv_dilations=(1, 2),
        max_seq_len=64,
        use_rotary_embeddings=True,
    )
    backbone = TemporalBackbone(config)
    x = torch.randn(3, 32, 8)
    out = backbone(x)
    assert out.shape == (3, 32, 32)


def test_temporal_backbone_cache_extends():
    config = TemporalBackboneConfig(
        feature_dim=4,
        model_dim=16,
        depth=1,
        heads=4,
        dropout=0.0,
        conv_kernel_size=3,
        conv_dilations=(1,),
        max_seq_len=128,
        use_rotary_embeddings=True,
    )
    backbone = TemporalBackbone(config)
    first = torch.randn(2, 32, 4)
    second = torch.randn(2, 96, 4)
    _ = backbone(first)
    out = backbone(second)
    assert out.shape == (2, 96, 16)


def test_temporal_backbone_no_rotary():
    config = TemporalBackboneConfig(
        feature_dim=6,
        model_dim=24,
        depth=1,
        heads=3,
        dropout=0.0,
        conv_kernel_size=3,
        conv_dilations=(1,),
        max_seq_len=32,
        use_rotary_embeddings=False,
    )
    backbone = TemporalBackbone(config)
    x = torch.randn(4, 16, 6)
    out = backbone(x)
    assert out.shape == (4, 16, 24)
