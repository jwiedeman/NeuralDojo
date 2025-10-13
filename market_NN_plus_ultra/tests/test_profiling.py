from market_nn_plus_ultra.training.config import ModelConfig
from market_nn_plus_ultra.utils import profile_backbone_throughput


def test_profile_backbone_throughput_cpu():
    config = ModelConfig(
        feature_dim=16,
        model_dim=64,
        depth=2,
        heads=4,
        dropout=0.1,
        conv_kernel_size=3,
        conv_dilations=(1, 2),
        architecture="temporal_transformer",
        horizon=3,
    )
    report = profile_backbone_throughput(
        config,
        batch_size=2,
        seq_len=32,
        warmup_steps=1,
        measure_steps=2,
    )
    assert report.tokens_per_second > 0
    assert report.seconds_per_step > 0
    assert report.architecture == config.architecture
    assert report.to_dict()["device"] == "cpu"
