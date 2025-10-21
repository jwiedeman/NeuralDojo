from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from market_nn_plus_ultra.evaluation import (
    MilestoneReference,
    generate_html_report,
    generate_markdown_report,
    generate_report,
)


matplotlib.use("Agg")


def _sample_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    timestamps = pd.date_range("2024-01-01", periods=64, freq="h")
    returns = rng.normal(0.001, 0.01, size=64)
    symbols = ["BTC", "ETH"] * 32
    data = {
        "symbol": symbols,
        "window_end": timestamps,
        "realised_return": returns,
    }
    return pd.DataFrame(data)


def test_generate_markdown_report(tmp_path: Path) -> None:
    predictions = _sample_predictions()
    output = tmp_path / "report.md"
    report_path = generate_markdown_report(predictions, output, title="Test Report")

    assert report_path.exists()
    contents = report_path.read_text(encoding="utf-8")
    assert "Test Report" in contents
    assert "Sharpe" in contents
    assert "## Profitability Summary" in contents
    assert "## Attribution by Symbol" in contents
    assert "## Scenario Analysis" in contents
    assert "## Bootstrapped Confidence Intervals" in contents

    assets_dir = tmp_path / "report_assets"
    assert assets_dir.exists()
    assert any(p.name.endswith(".png") for p in assets_dir.iterdir())


def test_markdown_includes_milestones(tmp_path: Path) -> None:
    predictions = _sample_predictions()
    output = tmp_path / "milestones.md"
    milestone = MilestoneReference(
        phase="Phase 3 — Evaluation & Monitoring",
        milestone="Publish automated run reports",
        summary="First automated alignment with research agenda milestones.",
    )

    report_path = generate_markdown_report(
        predictions,
        output,
        milestones=[milestone],
    )

    contents = report_path.read_text(encoding="utf-8")
    assert "## Research Agenda Alignment" in contents
    assert "Phase 3 — Evaluation & Monitoring" in contents
    assert "Publish automated run reports" in contents
    assert "First automated alignment" in contents


def test_generate_html_report(tmp_path: Path) -> None:
    predictions = _sample_predictions()
    output = tmp_path / "web.html"
    report_path = generate_html_report(predictions, output, title="HTML Report")

    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "HTML Report" in html
    assert "table" in html
    assert "<h2>Profitability Summary</h2>" in html
    assert "<h2>Attribution by Symbol</h2>" in html
    assert "<h2>Scenario Analysis</h2>" in html
    assert "<h2>Bootstrapped Confidence Intervals</h2>" in html


def test_html_includes_milestones(tmp_path: Path) -> None:
    predictions = _sample_predictions()
    output = tmp_path / "milestones.html"
    milestone = {
        "phase": "Phase 3 — Evaluation & Monitoring",
        "milestone": "Publish automated run reports",
        "summary": "HTML narrative includes research agenda context.",
    }

    report_path = generate_html_report(
        predictions,
        output,
        milestones=[milestone],
    )

    html = report_path.read_text(encoding="utf-8")
    assert "<h2>Research Agenda Alignment</h2>" in html
    assert "Phase 3 — Evaluation & Monitoring" in html
    assert "Publish automated run reports" in html
    assert "HTML narrative includes research agenda context." in html


def test_generate_report_infers_format(tmp_path: Path) -> None:
    predictions = _sample_predictions()
    md_path = tmp_path / "auto.md"
    html_path = tmp_path / "auto.html"

    out1 = generate_report(predictions, md_path)
    out2 = generate_report(predictions, html_path)

    assert out1.suffix == ".md"
    assert out2.suffix == ".html"


def test_custom_charts_dir_name(tmp_path: Path) -> None:
    predictions = _sample_predictions()
    output = tmp_path / "report.md"
    charts_dir = "custom_assets"

    report_path = generate_markdown_report(
        predictions,
        output,
        charts_dir_name=charts_dir,
    )

    assert report_path.exists()
    custom_assets = tmp_path / charts_dir
    assert custom_assets.exists()
    assert any(p.suffix == ".png" for p in custom_assets.iterdir())
    default_assets = tmp_path / "report_assets"
    assert not default_assets.exists()
