from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from market_nn_plus_ultra.evaluation.benchmarking import (
    architecture_leaderboard,
    dataframe_to_markdown,
    format_markdown_table,
    load_benchmark_frames,
    summarise_architecture_performance,
    summaries_to_frame,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "architecture": ["omni", "omni", "hybrid"],
            "label": ["omni-small", "omni-large", "hybrid"],
            "dataset_universe": ["equities", "crypto", "equities"],
            "metric_val_loss": [0.12, 0.10, 0.15],
            "duration_seconds": [12.0, 20.0, 8.0],
            "profitability_roi": [0.05, 0.08, 0.04],
        }
    )


def test_summarise_architecture_performance_orders_by_best_metric() -> None:
    frame = _sample_frame()
    summaries = summarise_architecture_performance(frame)
    assert [summary.architecture for summary in summaries] == ["omni", "hybrid"]

    omni_summary = summaries[0]
    assert omni_summary.scenario_count == 2
    assert omni_summary.best_metric == pytest.approx(0.10)
    assert omni_summary.best_metric_label == "omni-large"
    assert omni_summary.median_metric == pytest.approx(0.11)
    assert omni_summary.mean_duration == pytest.approx(16.0)
    assert omni_summary.mean_profitability == pytest.approx(0.065)


def test_format_markdown_table_renders_expected_structure() -> None:
    frame = _sample_frame()
    summaries = summarise_architecture_performance(frame)
    markdown = format_markdown_table(summaries, metric="metric_val_loss")
    assert "| Architecture |" in markdown
    assert "omni" in markdown
    assert "hybrid" in markdown


def test_load_benchmark_frames_supports_multiple_formats(tmp_path: Path) -> None:
    frame = _sample_frame()
    csv_path = tmp_path / "bench.csv"
    json_path = tmp_path / "bench.json"
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records")

    combined = load_benchmark_frames([csv_path, json_path])
    assert combined.shape[0] == frame.shape[0] * 2
    assert set(combined["__source"]) == {"bench.csv", "bench.json"}


def _load_summary_cli() -> object:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "summarise_architecture_sweep.py"
    spec = importlib.util.spec_from_file_location("summarise_architecture_sweep", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_cli_writes_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_summary_cli()
    frame = _sample_frame()
    input_path = tmp_path / "benchmark.csv"
    frame.to_csv(input_path, index=False)

    output_path = tmp_path / "summary.md"
    exit_code = module.main([
        str(input_path),
        "--output",
        str(output_path),
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "| Architecture |" in captured.out
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "omni" in text
    assert "hybrid" in text


def test_summary_cli_leaderboard_support(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_summary_cli()
    frame = _sample_frame()
    input_path = tmp_path / "benchmark.csv"
    frame.to_csv(input_path, index=False)

    leaderboard_path = tmp_path / "leaderboard.md"
    exit_code = module.main(
        [
            str(input_path),
            "--leaderboard-group-by",
            "dataset_universe",
            "--leaderboard-top-k",
            "1",
            "--leaderboard-output",
            str(leaderboard_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Leaderboard" in captured.out
    assert leaderboard_path.exists()
    leaderboard_text = leaderboard_path.read_text(encoding="utf-8")
    assert "dataset_universe" in leaderboard_text


def test_summaries_to_frame_round_trips() -> None:
    frame = _sample_frame()
    summaries = summarise_architecture_performance(frame)
    summary_frame = summaries_to_frame(summaries)
    assert set(summary_frame.columns) >= {
        "architecture",
        "scenario_count",
        "best_metric",
        "median_metric",
    }
    assert summary_frame.shape[0] == 2


def test_architecture_leaderboard_groups_and_ranks() -> None:
    frame = pd.DataFrame(
        {
            "architecture": ["omni", "hybrid", "omni", "state"],
            "label": ["omni-eq", "hybrid-eq", "omni-crypto", "state-crypto"],
            "dataset_universe": ["equities", "equities", "crypto", "crypto"],
            "metric_val_loss": [0.1, 0.12, 0.2, 0.18],
            "duration_seconds": [12.0, 8.0, 16.0, 10.0],
        }
    )

    leaderboard = architecture_leaderboard(
        frame,
        metric="metric_val_loss",
        higher_is_better=False,
        group_by=("dataset_universe",),
        top_k=2,
    )

    assert set(leaderboard["dataset_universe"]) == {"equities", "crypto"}
    equities = leaderboard[leaderboard["dataset_universe"] == "equities"].sort_values("rank")
    assert equities.iloc[0]["architecture"] == "omni"
    assert equities.iloc[0]["rank"] == 1
    assert equities.iloc[0]["metric_val_loss"] == pytest.approx(0.1)
    assert equities.iloc[1]["architecture"] == "hybrid"
    assert equities.iloc[1]["rank"] == 2


def test_dataframe_to_markdown_renders_expected_headers() -> None:
    frame = pd.DataFrame(
        {
            "group": ["equities"],
            "rank": [1],
            "architecture": ["omni"],
            "metric_val_loss": [0.123456],
        }
    )
    markdown = dataframe_to_markdown(frame)
    assert "| group |" in markdown
    assert "omni" in markdown
