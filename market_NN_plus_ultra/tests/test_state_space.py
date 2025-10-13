import torch

from market_nn_plus_ultra.models.state_space import (
    StateSpaceBackbone,
    StateSpaceConfig,
    initialise_state_space_backbone,
)


def test_state_space_forward_shape() -> None:
    config = StateSpaceConfig(feature_dim=8, model_dim=32, depth=3, state_dim=16, kernel_size=5)
    model = StateSpaceBackbone(config)
    batch = torch.randn(2, 20, 8)
    output = model(batch)
    assert output.shape == (2, 20, 32)


def test_initialise_state_space_backbone_sets_parameters() -> None:
    backbone = initialise_state_space_backbone(8, model_dim=16, depth=2, state_dim=8, kernel_size=3)
    total_params = sum(p.numel() for p in backbone.parameters())
    assert total_params > 0
