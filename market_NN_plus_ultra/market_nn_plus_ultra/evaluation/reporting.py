"""Reporting utilities for Market NN Plus Ultra evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import compute_equity_curve, risk_metrics
from ..utils.reporting import sanitize_metrics


@dataclass(slots=True)
class ReportSummary:
    """Metadata describing the predictions underpinning a report."""

    num_rows: int
    start_timestamp: str | None
    end_timestamp: str | None
    symbols: int


def _coerce_returns(predictions: pd.DataFrame, return_column: str) -> np.ndarray:
    if return_column not in predictions.columns:
        raise ValueError(f"Column '{return_column}' not present in predictions frame")
    numeric = pd.to_numeric(predictions[return_column], errors="coerce")
    return numeric.to_numpy(dtype=np.float64)


def _summarise(predictions: pd.DataFrame) -> ReportSummary:
    num_rows = int(len(predictions))
    start = None
    end = None
    if "window_end" in predictions.columns and not predictions.empty:
        timestamps = pd.to_datetime(predictions["window_end"], errors="coerce").dropna().sort_values()
        if not timestamps.empty:
            start = timestamps.iloc[0].isoformat()
            end = timestamps.iloc[-1].isoformat()
    symbols = int(predictions.get("symbol", pd.Series(dtype="object")).nunique())
    return ReportSummary(num_rows=num_rows, start_timestamp=start, end_timestamp=end, symbols=symbols)


def _format_metric(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.6f}"


def _prepare_report(
    predictions: pd.DataFrame,
    *,
    return_column: str,
    metrics: Optional[Mapping[str, float]],
    periods_per_year: int,
) -> tuple[np.ndarray, ReportSummary, Dict[str, float]]:
    returns = _coerce_returns(predictions, return_column)
    summary = _summarise(predictions)
    computed = sanitize_metrics(risk_metrics(returns, periods_per_year=periods_per_year))
    if metrics is not None:
        merged = dict(computed)
        merged.update(metrics)
        final_metrics = sanitize_metrics(merged)
    else:
        final_metrics = computed
    return returns, summary, final_metrics


def _generate_charts(
    returns: np.ndarray,
    charts_dir: Path,
    *,
    include_equity: bool,
    include_distribution: bool,
    periods_per_year: int,
) -> Dict[str, Path]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts: Dict[str, Path] = {}

    cleaned = returns[~np.isnan(returns)] if returns.size else returns
    if include_equity and cleaned.size:
        equity = compute_equity_curve(cleaned)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(equity, color="#2E86AB", linewidth=2)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Step")
        ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.3)
        path = charts_dir / "equity_curve.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        charts["equity_curve"] = path

    if include_distribution and cleaned.size:
        annualised = cleaned * np.sqrt(periods_per_year)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(cleaned, bins=40, alpha=0.8, label="Per-period")
        ax.hist(annualised, bins=40, alpha=0.4, label="Annualised")
        ax.set_title("Return Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = charts_dir / "return_distribution.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        charts["return_distribution"] = path

    return charts


def _build_markdown(
    *,
    title: str,
    description: str | None,
    summary: ReportSummary,
    metrics: Mapping[str, float],
    charts: Mapping[str, Path],
    output_dir: Path,
) -> str:
    lines: list[str] = [f"# {title}"]
    if description:
        lines += ["", description.strip(), ""]

    lines += [
        "## Overview",
        "",
        f"- **Total samples:** {summary.num_rows}",
        f"- **Symbols:** {summary.symbols}",
    ]
    if summary.start_timestamp:
        lines.append(f"- **Window:** {summary.start_timestamp} → {summary.end_timestamp}")
    lines.append("")

    lines += ["## Risk Metrics", "", "| Metric | Value |", "| --- | --- |"]
    for name in sorted(metrics.keys()):
        lines.append(f"| {name.replace('_', ' ').title()} | {_format_metric(metrics[name])} |")
    lines.append("")

    if charts:
        lines.append("## Visualisations")
        lines.append("")
        for label, path in charts.items():
            rel = path.relative_to(output_dir)
            pretty = label.replace("_", " ").title()
            lines.append(f"![{pretty}]({rel.as_posix()})")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_html(
    *,
    title: str,
    description: str | None,
    summary: ReportSummary,
    metrics: Mapping[str, float],
    charts: Mapping[str, Path],
    output_dir: Path,
) -> str:
    overview_items = [
        f"<li><strong>Total samples:</strong> {summary.num_rows}</li>",
        f"<li><strong>Symbols:</strong> {summary.symbols}</li>",
    ]
    if summary.start_timestamp:
        overview_items.append(
            f"<li><strong>Window:</strong> {summary.start_timestamp} → {summary.end_timestamp}</li>"
        )

    metrics_rows = "\n".join(
        f"<tr><td>{name.replace('_', ' ').title()}</td><td>{_format_metric(value)}</td></tr>"
        for name, value in sorted(metrics.items())
    )

    figures = "\n".join(
        f"<figure><img src=\"{path.relative_to(output_dir).as_posix()}\" alt=\"{label}\">"
        f"<figcaption>{label.replace('_', ' ').title()}</figcaption></figure>"
        for label, path in charts.items()
    )

    description_html = f"<p>{description.strip()}</p>" if description else ""

    body = "\n".join(
        [
            f"<h1>{title}</h1>",
            description_html,
            "<h2>Overview</h2>",
            f"<ul>{''.join(overview_items)}</ul>",
            "<h2>Risk Metrics</h2>",
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
            f"{metrics_rows}</tbody></table>",
            "<h2>Visualisations</h2>" if charts else "",
            figures,
        ]
    )

    style = (
        "body{font-family:Arial,sans-serif;margin:2rem;}"
        "h1,h2{color:#1f3c5b;}"
        "table{border-collapse:collapse;margin:1.5rem 0;min-width:320px;}"
        "table,th,td{border:1px solid #d0d7de;padding:0.6rem;}th{background:#f6f8fa;}"
        "ul{list-style:disc;margin-left:1.5rem;}"
        "figure{margin:1.5rem 0;}figcaption{text-align:center;color:#555;font-size:0.9rem;}"
        "img{max-width:100%;height:auto;box-shadow:0 1px 4px rgba(0,0,0,0.1);}"
    )

    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            f"  <title>{title}</title>",
            f"  <style>{style}</style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
    )


def generate_markdown_report(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    metrics: Optional[Mapping[str, float]] = None,
    return_column: str = "realised_return",
    title: str = "Market NN Plus Ultra Performance Report",
    description: str | None = None,
    include_equity_chart: bool = True,
    include_distribution_chart: bool = True,
    periods_per_year: int = 252,
    charts_dir_name: str | None = None,
) -> Path:
    """Write a Markdown performance report and return the file path."""

    output = Path(output_path)
    if not output.suffix:
        output = output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)

    returns, summary, final_metrics = _prepare_report(
        predictions,
        return_column=return_column,
        metrics=metrics,
        periods_per_year=periods_per_year,
    )

    charts_dir = output.parent / (
        charts_dir_name if charts_dir_name else f"{output.stem}_assets"
    )
    charts = _generate_charts(
        returns,
        charts_dir,
        include_equity=include_equity_chart,
        include_distribution=include_distribution_chart,
        periods_per_year=periods_per_year,
    )

    markdown_text = _build_markdown(
        title=title,
        description=description,
        summary=summary,
        metrics=final_metrics,
        charts=charts,
        output_dir=output.parent,
    )
    output.write_text(markdown_text, encoding="utf-8")
    return output


def generate_html_report(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    metrics: Optional[Mapping[str, float]] = None,
    return_column: str = "realised_return",
    title: str = "Market NN Plus Ultra Performance Report",
    description: str | None = None,
    include_equity_chart: bool = True,
    include_distribution_chart: bool = True,
    periods_per_year: int = 252,
    charts_dir_name: str | None = None,
) -> Path:
    """Write an HTML performance report and return the file path."""

    output = Path(output_path)
    if not output.suffix:
        output = output.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)

    returns, summary, final_metrics = _prepare_report(
        predictions,
        return_column=return_column,
        metrics=metrics,
        periods_per_year=periods_per_year,
    )

    charts_dir = output.parent / (
        charts_dir_name if charts_dir_name else f"{output.stem}_assets"
    )
    charts = _generate_charts(
        returns,
        charts_dir,
        include_equity=include_equity_chart,
        include_distribution=include_distribution_chart,
        periods_per_year=periods_per_year,
    )

    html_text = _build_html(
        title=title,
        description=description,
        summary=summary,
        metrics=final_metrics,
        charts=charts,
        output_dir=output.parent,
    )
    output.write_text(html_text, encoding="utf-8")
    return output


def generate_report(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    metrics: Optional[Mapping[str, float]] = None,
    return_column: str = "realised_return",
    title: str = "Market NN Plus Ultra Performance Report",
    description: str | None = None,
    include_equity_chart: bool = True,
    include_distribution_chart: bool = True,
    periods_per_year: int = 252,
    charts_dir_name: str | None = None,
) -> Path:
    """Generate a performance report; format inferred from suffix."""

    output = Path(output_path)
    suffix = output.suffix.lower()
    if suffix == ".html":
        return generate_html_report(
            predictions,
            output,
            metrics=metrics,
            return_column=return_column,
            title=title,
            description=description,
            include_equity_chart=include_equity_chart,
            include_distribution_chart=include_distribution_chart,
            periods_per_year=periods_per_year,
            charts_dir_name=charts_dir_name,
        )
    return generate_markdown_report(
        predictions,
        output if suffix else output.with_suffix(".md"),
        metrics=metrics,
        return_column=return_column,
        title=title,
        description=description,
        include_equity_chart=include_equity_chart,
        include_distribution_chart=include_distribution_chart,
        periods_per_year=periods_per_year,
        charts_dir_name=charts_dir_name,
    )


__all__ = [
    "ReportSummary",
    "generate_markdown_report",
    "generate_html_report",
    "generate_report",
]
