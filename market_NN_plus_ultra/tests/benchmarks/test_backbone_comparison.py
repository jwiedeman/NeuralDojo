from datetime import datetime
from pathlib import Path

from market_nn_plus_ultra.data import FixtureConfig
from market_nn_plus_ultra.evaluation.backbone_comparison import (
    AssetUniverseSpec,
    generate_backbone_report,
    run_backbone_comparison,
)
from market_nn_plus_ultra.training import BenchmarkScenario, TrainerOverrides


def test_run_backbone_comparison_generates_results(tmp_path: Path) -> None:
    universe = AssetUniverseSpec(
        name="mini",
        fixture=FixtureConfig(
            symbols=["EQ1", "EQ2"],
            rows=96,
            freq="15min",
            seed=77,
            start=datetime(2024, 1, 1),
            alt_features=0,
        ),
    )

    overrides = TrainerOverrides(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        batch_size=4,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
    )

    scenarios = [
        BenchmarkScenario(
            architecture="hybrid_transformer",
            model_dim=32,
            depth=1,
            horizon=2,
            conv_dilations=(1, 2),
            label="hybrid",
        ),
        BenchmarkScenario(
            architecture="omni_mixture",
            model_dim=32,
            depth=1,
            horizon=2,
            conv_dilations=(1, 2),
            label="omni",
        ),
    ]

    frame = run_backbone_comparison(
        [universe],
        scenarios=scenarios,
        trainer_overrides=overrides,
        window_size=24,
        horizon=2,
        stride=4,
        base_model_dim=32,
        base_depth=1,
        base_batch_size=4,
        regenerate_fixtures=True,
        fixture_root=tmp_path / "fixtures",
    )

    assert set(frame["architecture"]) == {"hybrid_transformer", "omni_mixture"}
    assert frame["universe"].unique().tolist() == [universe.display_name]
    assert "metric_val_loss" in frame.columns

    report = generate_backbone_report(frame)
    assert "| architecture |" in report
