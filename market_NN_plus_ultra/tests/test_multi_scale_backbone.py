import torch

from market_nn_plus_ultra.models.multi_scale import MultiScaleBackbone, MultiScaleBackboneConfig


def test_multi_scale_backbone_preserves_sequence_length():
    config = MultiScaleBackboneConfig(
        feature_dim=16,
        model_dim=32,
        scales=(1, 4, 8),
        depth_per_scale=2,
        conv_dilations=(1, 2),
        max_seq_len=256,
    )
    backbone = MultiScaleBackbone(config)
    inputs = torch.randn(3, 120, 16)
    outputs = backbone(inputs)
    assert outputs.shape == (3, 120, 32)


def test_multi_scale_backbone_handles_irregular_lengths():
    config = MultiScaleBackboneConfig(
        feature_dim=8,
        model_dim=16,
        scales=(1, 5),
        depth_per_scale=1,
        conv_dilations=(1,),
        max_seq_len=128,
    )
    backbone = MultiScaleBackbone(config)
    inputs = torch.randn(2, 73, 8)
    outputs = backbone(inputs)
    assert outputs.shape == (2, 73, 16)


def test_multi_scale_fusion_weights_form_probability_simplex():
    config = MultiScaleBackboneConfig(
        feature_dim=4,
        model_dim=12,
        heads=3,
        scales=(1, 2, 4),
        depth_per_scale=1,
        conv_dilations=(1,),
        max_seq_len=64,
    )
    backbone = MultiScaleBackbone(config)
    logits = backbone.fusion.gates
    weights = torch.softmax(logits, dim=0)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
    assert torch.all(weights > 0)
