import torch

from market_nn_plus_ultra.training.config import ModelConfig, OptimizerConfig
from market_nn_plus_ultra.training.train_loop import MarketLightningModule


def test_validation_accumulates_profitability_summary() -> None:
    model_cfg = ModelConfig(
        feature_dim=4,
        model_dim=16,
        depth=2,
        heads=2,
        horizon=2,
        output_dim=1,
        architecture="temporal_transformer",
        conv_kernel_size=3,
        conv_dilations=(1, 2, 4, 8),
    )
    module = MarketLightningModule(model_cfg, OptimizerConfig())
    module.log = lambda *args, **kwargs: None  # type: ignore[assignment]

    batch = {
        "features": torch.randn(3, 32, model_cfg.feature_dim),
        "targets": torch.exp(torch.randn(3, model_cfg.horizon, 1)),
        "reference": torch.exp(torch.randn(3, 1)),
    }

    module.on_validation_epoch_start()
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()

    summary = module.latest_profitability_summary
    assert summary is not None
    assert set(summary.keys()) == {"roi", "sharpe", "max_drawdown"}
    assert all(isinstance(value, float) for value in summary.values())
