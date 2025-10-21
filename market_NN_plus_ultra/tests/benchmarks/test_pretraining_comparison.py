import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_nn_plus_ultra.training import TrainingRunResult


def _load_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmarks"
        / "pretraining_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("pretraining_comparison", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretraining_comparison_runs_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    module = _load_module()

    trainer_stub = SimpleNamespace(checkpoint_dir=tmp_path / "base")
    base_config = SimpleNamespace(
        trainer=trainer_stub,
        wandb_project="plus-ultra",
        wandb_entity="neuraldojo",
        wandb_run_name="baseline",
        wandb_tags=("baseline",),
        wandb_offline=False,
    )

    calls: list[tuple[Path | None, Path]] = []

    def fake_load_experiment(_path: Path):
        return base_config

    def fake_run_training(config, *, pretrain_checkpoint_path=None):
        calls.append((
            Path(pretrain_checkpoint_path) if pretrain_checkpoint_path is not None else None,
            Path(config.trainer.checkpoint_dir),
        ))
        return TrainingRunResult(
            best_model_path=str(Path(config.trainer.checkpoint_dir) / "model.ckpt"),
            logged_metrics={"val/loss": 0.1 if pretrain_checkpoint_path is None else 0.08},
            dataset_summary={"train_windows": 8, "val_windows": 4},
            profitability_summary={"roi": 0.5 if pretrain_checkpoint_path is None else 0.65},
        )

    monkeypatch.setattr(module, "load_experiment_from_file", fake_load_experiment)
    monkeypatch.setattr(module, "run_training", fake_run_training)

    checkpoint_path = tmp_path / "pretrain.ckpt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    output_path = tmp_path / "results.parquet"
    workdir = tmp_path / "work"

    exit_code = module.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--pretrain-checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--workdir",
            str(workdir),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert len(calls) == 2

    scratch_call, warm_call = calls
    assert scratch_call[0] is None
    assert scratch_call[1].name == "scratch"
    assert warm_call[0] == checkpoint_path
    assert warm_call[1].name == "pretrained"

    frame = pd.read_parquet(output_path)
    assert sorted(frame["run"].tolist()) == [
        "pretrained",
        "pretrained_minus_scratch",
        "scratch",
    ]
    assert "metric_val_loss" in frame.columns
    delta = frame.set_index("run").loc["pretrained_minus_scratch"]
    assert pytest.approx(delta["metric_val_loss"], rel=1e-6) == -0.02
    assert pytest.approx(delta["profitability_roi"], rel=1e-6) == 0.15

    captured = capsys.readouterr().out
    assert "Warm-start vs scratch deltas" in captured

