"""Reporting utilities for Market NN Plus Ultra evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import compute_equity_curve, max_drawdown, risk_metrics
from ..utils.reporting import sanitize_metrics


@dataclass(slots=True)
class ReportSummary:
    """Metadata describing the predictions underpinning a report."""

    num_rows: int
    start_timestamp: str | None
    end_timestamp: str | None
    symbols: int


@dataclass(slots=True)
class MilestoneReference:
    """Reference to a research agenda milestone included in a report."""

    phase: str
    milestone: str
    summary: str | None = None

    def to_markdown(self) -> str:
        details = f": {self.summary}" if self.summary else ""
        return f"- **{self.phase} — {self.milestone}**{details}"

    def to_html(self) -> str:
        details = f": {self.summary}" if self.summary else ""
        return (
            "<li>"
            f"<strong>{self.phase}</strong> — {self.milestone}{details}"
            "</li>"
        )


@dataclass(slots=True)
class ProfitabilitySummary:
    """Structured profitability snapshot for the run."""

    total_return: float
    cumulative_return: float
    annualised_return: float
    annualised_volatility: float
    return_multiple: float
    hit_rate: float
    max_drawdown: float
    best_period: float
    worst_period: float
    positive_periods: int
    negative_periods: int

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("Total return", _format_percent(self.total_return)),
            ("Cumulative PnL", f"{self.cumulative_return:.6f}"),
            ("Annualised return", _format_percent(self.annualised_return)),
            ("Annualised volatility", _format_percent(self.annualised_volatility)),
            ("Return multiple", f"{self.return_multiple:.4f}×"),
            ("Hit rate", _format_percent(self.hit_rate)),
            ("Max drawdown", _format_percent(abs(self.max_drawdown))),
            ("Best period", _format_percent(self.best_period)),
            ("Worst period", _format_percent(self.worst_period)),
            ("Positive periods", str(self.positive_periods)),
            ("Negative periods", str(self.negative_periods)),
        ]


@dataclass(slots=True)
class AttributionRow:
    """Per-symbol attribution record."""

    symbol: str
    sample_weight: float
    average_return: float
    cumulative_return: float
    contribution: float

    def markdown_row(self) -> str:
        return (
            f"| {self.symbol} | {_format_percent(self.sample_weight)} | "
            f"{_format_percent(self.average_return)} | {self.cumulative_return:.6f} | "
            f"{_format_percent(self.contribution)} |"
        )

    def html_row(self) -> str:
        return (
            "<tr>"
            f"<td>{self.symbol}</td>"
            f"<td>{_format_percent(self.sample_weight)}</td>"
            f"<td>{_format_percent(self.average_return)}</td>"
            f"<td>{self.cumulative_return:.6f}</td>"
            f"<td>{_format_percent(self.contribution)}</td>"
            "</tr>"
        )


@dataclass(slots=True)
class RegimeAttributionRow:
    """Attribution record describing performance for a regime bucket."""

    label: str
    samples: int
    sample_weight: float
    average_return: float
    cumulative_return: float
    sharpe: float
    hit_rate: float

    def markdown_row(self) -> str:
        return (
            f"| {self.label} | {self.samples} | {_format_percent(self.sample_weight)} | "
            f"{_format_percent(self.average_return)} | {_format_percent(self.cumulative_return)} | "
            f"{self.sharpe:.3f} | {_format_percent(self.hit_rate)} |"
        )

    def html_row(self) -> str:
        return (
            "<tr>"
            f"<td>{self.label}</td>"
            f"<td>{self.samples}</td>"
            f"<td>{_format_percent(self.sample_weight)}</td>"
            f"<td>{_format_percent(self.average_return)}</td>"
            f"<td>{_format_percent(self.cumulative_return)}</td>"
            f"<td>{self.sharpe:.3f}</td>"
            f"<td>{_format_percent(self.hit_rate)}</td>"
            "</tr>"
        )


@dataclass(slots=True)
class RegimeAttributionGroup:
    """Collection of regime attribution rows for a single regime signal."""

    name: str
    rows: Sequence[RegimeAttributionRow]


@dataclass(slots=True)
class ScenarioEvent:
    """A notable scenario (best/worst period)."""

    timestamp: str | None
    symbol: str | None
    return_value: float

    def markdown_row(self) -> str:
        timestamp = self.timestamp or "—"
        symbol = self.symbol or "—"
        return f"| {timestamp} | {symbol} | {_format_percent(self.return_value)} |"

    def html_row(self) -> str:
        timestamp = self.timestamp or "—"
        symbol = self.symbol or "—"
        return (
            "<tr>"
            f"<td>{timestamp}</td>"
            f"<td>{symbol}</td>"
            f"<td>{_format_percent(self.return_value)}</td>"
            "</tr>"
        )


@dataclass(slots=True)
class DrawdownEvent:
    """Largest drawdown observed in the backtest."""

    depth: float
    start_timestamp: str | None
    end_timestamp: str | None
    duration: int

    def markdown_line(self) -> str:
        start = self.start_timestamp or "—"
        end = self.end_timestamp or "—"
        return (
            "- **Max drawdown:** "
            f"{_format_percent(abs(self.depth))} from {start} to {end} "
            f"({self.duration} steps)"
        )

    def html_block(self) -> str:
        start = self.start_timestamp or "—"
        end = self.end_timestamp or "—"
        return (
            "<p><strong>Max drawdown:</strong> "
            f"{_format_percent(abs(self.depth))} from {start} to {end} "
            f"({self.duration} steps)</p>"
        )


@dataclass(slots=True)
class ScenarioAnalysis:
    """Scenario analysis results for the run."""

    best_periods: Sequence[ScenarioEvent]
    worst_periods: Sequence[ScenarioEvent]
    drawdown: DrawdownEvent | None


@dataclass(slots=True)
class ConfidenceInterval:
    """Bootstrapped confidence interval."""

    lower: float
    upper: float
    as_percentage: bool = False

    def format_markdown(self) -> str:
        return self._format_interval()

    def format_html(self) -> str:
        return self._format_interval()

    def _format_interval(self) -> str:
        if self.as_percentage:
            return f"[{self.lower:.2%}, {self.upper:.2%}]"
        return f"[{self.lower:.4f}, {self.upper:.4f}]"


@dataclass(slots=True)
class PreparedReport:
    """Container for all report artefacts."""

    returns: np.ndarray
    summary: ReportSummary
    metrics: Dict[str, float]
    profitability: ProfitabilitySummary
    attribution: Sequence[AttributionRow]
    regime_attribution: Sequence[RegimeAttributionGroup]
    scenarios: ScenarioAnalysis
    confidence_intervals: Mapping[str, ConfidenceInterval]


def _format_percent(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "∞%" if value > 0 else "-∞%"
    return f"{value:.2%}"


def _extract_return_series(predictions: pd.DataFrame, return_column: str) -> pd.Series:
    if return_column not in predictions.columns:
        raise ValueError(f"Column '{return_column}' not present in predictions frame")
    numeric = pd.to_numeric(predictions[return_column], errors="coerce")
    numeric.name = return_column
    return numeric


def _extract_optional_series(
    predictions: pd.DataFrame, column: str | None, *, label: str
) -> pd.Series | None:
    if column is None:
        return None
    if column not in predictions.columns:
        raise ValueError(f"Column '{column}' not present in predictions frame for {label}")
    numeric = pd.to_numeric(predictions[column], errors="coerce")
    numeric.name = column
    return numeric


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


def _compute_profitability(
    returns_series: pd.Series,
    metrics: Mapping[str, float],
) -> ProfitabilitySummary:
    cleaned = returns_series.dropna()
    np_returns = cleaned.to_numpy(dtype=np.float64)
    best_period = float(np.max(np_returns)) if np_returns.size else 0.0
    worst_period = float(np.min(np_returns)) if np_returns.size else 0.0
    positive = int((np_returns > 0).sum()) if np_returns.size else 0
    negative = int((np_returns < 0).sum()) if np_returns.size else 0

    return ProfitabilitySummary(
        total_return=float(metrics.get("total_return", 0.0)),
        cumulative_return=float(metrics.get("cumulative_return", 0.0)),
        annualised_return=float(metrics.get("annualised_return", 0.0)),
        annualised_volatility=float(metrics.get("volatility", 0.0)),
        return_multiple=float(metrics.get("return_multiple", 0.0)),
        hit_rate=float(metrics.get("hit_rate", 0.0)),
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        best_period=best_period,
        worst_period=worst_period,
        positive_periods=positive,
        negative_periods=negative,
    )


def _compute_attribution(
    predictions: pd.DataFrame,
    returns_series: pd.Series,
) -> list[AttributionRow]:
    if "symbol" not in predictions.columns:
        return []

    cleaned = returns_series.dropna()
    if cleaned.empty:
        return []

    frame = predictions.reindex(cleaned.index).copy()
    frame["__returns"] = cleaned
    grouped = (
        frame.groupby("symbol", observed=True)["__returns"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "average", "sum": "cumulative", "count": "samples"})
    )

    total_cumulative = float(grouped["cumulative"].sum())
    total_samples = int(grouped["samples"].sum()) or 1
    rows: list[AttributionRow] = []
    for symbol, record in grouped.sort_values("cumulative", ascending=False).iterrows():
        cumulative = float(record["cumulative"])
        rows.append(
            AttributionRow(
                symbol=str(symbol),
                sample_weight=float(record["samples"]) / total_samples,
                average_return=float(record["average"]),
                cumulative_return=cumulative,
                contribution=(cumulative / total_cumulative) if total_cumulative else 0.0,
            )
        )
    return rows


def _compute_regime_attribution(
    predictions: pd.DataFrame,
    returns_series: pd.Series,
    *,
    periods_per_year: int,
) -> list[RegimeAttributionGroup]:
    regime_columns = [col for col in predictions.columns if col.startswith("regime__")]
    if not regime_columns:
        return []

    cleaned_returns = returns_series.dropna()
    total_samples = int(cleaned_returns.size)
    if total_samples == 0:
        return []

    groups: list[RegimeAttributionGroup] = []
    base_frame = predictions.reindex(cleaned_returns.index).copy()
    base_frame["__returns"] = cleaned_returns

    for column in sorted(regime_columns):
        regime_values = base_frame[column].dropna()
        if regime_values.empty:
            continue

        joined = base_frame.loc[regime_values.index, ["__returns", column]].dropna()
        if joined.empty:
            continue

        rows: list[RegimeAttributionRow] = []
        for label, group in joined.groupby(column, observed=True):
            returns = group["__returns"].to_numpy(dtype=np.float64)
            if returns.size == 0:
                continue
            metrics = risk_metrics(returns, periods_per_year=periods_per_year)
            rows.append(
                RegimeAttributionRow(
                    label=str(label),
                    samples=int(returns.size),
                    sample_weight=(returns.size / total_samples) if total_samples else 0.0,
                    average_return=float(metrics.get("average_return", float("nan"))),
                    cumulative_return=float(metrics.get("cumulative_return", float("nan"))),
                    sharpe=float(metrics.get("sharpe", float("nan"))),
                    hit_rate=float(metrics.get("hit_rate", float("nan"))),
                )
            )

        if rows:
            rows.sort(key=lambda row: row.sample_weight, reverse=True)
            base_name = column.split("regime__", 1)[1] or column
            groups.append(RegimeAttributionGroup(name=base_name, rows=rows))

    return groups


def _resolve_timestamp(row: pd.Series) -> str | None:
    for column in ("window_end", "timestamp", "time", "date"):
        if column in row and pd.notna(row[column]):
            try:
                return pd.to_datetime(row[column]).isoformat()
            except Exception:  # pragma: no cover - defensive fallback
                continue
    return None


def _compute_drawdown_event(
    returns_series: pd.Series,
    base_frame: pd.DataFrame,
) -> DrawdownEvent | None:
    cleaned = returns_series.dropna()
    if cleaned.empty:
        return None

    np_returns = cleaned.to_numpy(dtype=np.float64)
    equity = compute_equity_curve(np_returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    trough_idx = int(np.argmin(drawdowns))
    if trough_idx <= 0:
        return None
    peak_idx = int(np.argmax(equity[: trough_idx + 1]))

    def _timestamp_for_equity_index(idx: int) -> str | None:
        position = idx - 1
        if position < 0 or position >= len(base_frame):
            return None
        row = base_frame.iloc[position]
        return _resolve_timestamp(row)

    start_ts = _timestamp_for_equity_index(peak_idx)
    end_ts = _timestamp_for_equity_index(trough_idx)
    duration = max(trough_idx - peak_idx, 0)
    return DrawdownEvent(
        depth=float(drawdowns[trough_idx]),
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        duration=duration,
    )


def _compute_scenarios(
    predictions: pd.DataFrame,
    returns_series: pd.Series,
) -> ScenarioAnalysis:
    cleaned = returns_series.dropna()
    if cleaned.empty:
        return ScenarioAnalysis(best_periods=[], worst_periods=[], drawdown=None)

    frame = predictions.reindex(cleaned.index).copy()
    frame["__returns"] = cleaned
    best = frame.nlargest(5, "__returns")
    worst = frame.nsmallest(5, "__returns")

    best_events = [
        ScenarioEvent(
            timestamp=_resolve_timestamp(row),
            symbol=str(row.get("symbol")) if "symbol" in row else None,
            return_value=float(row["__returns"]),
        )
        for _, row in best.iterrows()
    ]
    worst_events = [
        ScenarioEvent(
            timestamp=_resolve_timestamp(row),
            symbol=str(row.get("symbol")) if "symbol" in row else None,
            return_value=float(row["__returns"]),
        )
        for _, row in worst.iterrows()
    ]

    drawdown = _compute_drawdown_event(cleaned, frame)
    return ScenarioAnalysis(best_periods=best_events, worst_periods=worst_events, drawdown=drawdown)


def _bootstrap_confidence_intervals(
    returns_series: pd.Series,
    *,
    periods_per_year: int,
    n_bootstrap: int = 1000,
    seed: int = 7_531_441,
) -> dict[str, ConfidenceInterval]:
    cleaned = returns_series.dropna()
    np_returns = cleaned.to_numpy(dtype=np.float64)
    if np_returns.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    samples = rng.choice(np_returns, size=(n_bootstrap, np_returns.size), replace=True)
    mean_returns = samples.mean(axis=1)
    total_returns = samples.sum(axis=1)
    annualised = mean_returns * periods_per_year
    std = samples.std(axis=1, ddof=1)
    sharpe = np.divide(
        mean_returns * np.sqrt(periods_per_year),
        std,
        out=np.zeros_like(mean_returns),
        where=std > 0,
    )

    def _ci(values: np.ndarray) -> tuple[float, float]:
        lower = float(np.quantile(values, 0.025))
        upper = float(np.quantile(values, 0.975))
        return lower, upper

    ann_lower, ann_upper = _ci(annualised)
    total_lower, total_upper = _ci(total_returns)
    sharpe_lower, sharpe_upper = _ci(sharpe)

    return {
        "annualised_return": ConfidenceInterval(ann_lower, ann_upper, as_percentage=True),
        "total_return": ConfidenceInterval(total_lower, total_upper, as_percentage=True),
        "sharpe_ratio": ConfidenceInterval(sharpe_lower, sharpe_upper, as_percentage=False),
    }


def _prepare_report(
    predictions: pd.DataFrame,
    *,
    return_column: str,
    benchmark_column: str | None,
    metrics: Optional[Mapping[str, float]],
    periods_per_year: int,
) -> PreparedReport:
    returns_series = _extract_return_series(predictions, return_column)
    benchmark_series = _extract_optional_series(
        predictions, benchmark_column, label="benchmark"
    )
    returns_array = returns_series.to_numpy(dtype=np.float64)
    summary = _summarise(predictions)

    if benchmark_series is not None:
        aligned = pd.concat(
            [
                returns_series.rename("__returns"),
                benchmark_series.rename("__benchmark"),
            ],
            axis=1,
        ).dropna()
        returns_for_metrics = aligned["__returns"].to_numpy(dtype=np.float64)
        benchmark_array = aligned["__benchmark"].to_numpy(dtype=np.float64)
    else:
        returns_for_metrics = returns_series.dropna().to_numpy(dtype=np.float64)
        benchmark_array = None

    computed = sanitize_metrics(
        risk_metrics(
            returns_for_metrics,
            periods_per_year=periods_per_year,
            benchmark_returns=benchmark_array,
        )
    )
    if metrics is not None:
        merged = dict(computed)
        merged.update(metrics)
        final_metrics = sanitize_metrics(merged)
    else:
        final_metrics = computed

    profitability = _compute_profitability(returns_series, final_metrics)
    attribution = _compute_attribution(predictions, returns_series)
    regime_attribution = _compute_regime_attribution(
        predictions,
        returns_series,
        periods_per_year=periods_per_year,
    )
    scenarios = _compute_scenarios(predictions, returns_series)
    confidence = _bootstrap_confidence_intervals(
        returns_series, periods_per_year=periods_per_year
    )

    return PreparedReport(
        returns=returns_array,
        summary=summary,
        metrics=final_metrics,
        profitability=profitability,
        attribution=attribution,
        regime_attribution=regime_attribution,
        scenarios=scenarios,
        confidence_intervals=confidence,
    )


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


def _normalise_milestones(
    milestones: Optional[Iterable[MilestoneReference | Mapping[str, str]]],
) -> list[MilestoneReference]:
    if not milestones:
        return []
    normalised: list[MilestoneReference] = []
    for item in milestones:
        if isinstance(item, MilestoneReference):
            normalised.append(item)
            continue
        phase = item.get("phase")
        milestone = item.get("milestone") or item.get("title") or item.get("name")
        if not phase or not milestone:
            raise ValueError(
                "Milestone references must include at least 'phase' and 'milestone' keys"
            )
        summary = item.get("summary") or item.get("notes")
        normalised.append(
            MilestoneReference(
                phase=str(phase),
                milestone=str(milestone),
                summary=str(summary) if summary is not None else None,
            )
        )
    return normalised


def _build_markdown(
    *,
    title: str,
    description: str | None,
    summary: ReportSummary,
    metrics: Mapping[str, float],
    charts: Mapping[str, Path],
    output_dir: Path,
    milestones: Sequence[MilestoneReference] = (),
    profitability: ProfitabilitySummary | None = None,
    attribution: Sequence[AttributionRow] = (),
    regime_attribution: Sequence[RegimeAttributionGroup] = (),
    scenarios: ScenarioAnalysis | None = None,
    confidence_intervals: Mapping[str, ConfidenceInterval] = (),
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

    if profitability is not None:
        lines.append("## Profitability Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for name, value in profitability.rows():
            lines.append(f"| {name} | {value} |")
        lines.append("")

    if milestones:
        lines.append("## Research Agenda Alignment")
        lines.append("")
        lines.extend(ms.to_markdown() for ms in milestones)
        lines.append("")

    if attribution:
        lines.append("## Attribution by Symbol")
        lines.append("")
        lines.append("| Symbol | Sample Weight | Average Return | Cumulative PnL | Contribution |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in attribution:
            lines.append(row.markdown_row())
        lines.append("")

    if regime_attribution:
        lines.append("## Regime Attribution")
        lines.append("")
        for group in regime_attribution:
            display = group.name.replace("_", " ").title()
            lines.append(f"### {display}")
            lines.append("")
            lines.append(
                "| Label | Samples | Sample Weight | Average Return | Cumulative Return | Sharpe | Hit Rate |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for row in group.rows:
                lines.append(row.markdown_row())
            lines.append("")

    if scenarios is not None and (
        scenarios.best_periods or scenarios.worst_periods or scenarios.drawdown
    ):
        lines.append("## Scenario Analysis")
        lines.append("")
        if scenarios.drawdown is not None:
            lines.append(scenarios.drawdown.markdown_line())
            lines.append("")
        if scenarios.best_periods:
            lines.append("### Best Periods")
            lines.append("")
            lines.append("| Timestamp | Symbol | Return |")
            lines.append("| --- | --- | --- |")
            for event in scenarios.best_periods:
                lines.append(event.markdown_row())
            lines.append("")
        if scenarios.worst_periods:
            lines.append("### Worst Periods")
            lines.append("")
            lines.append("| Timestamp | Symbol | Return |")
            lines.append("| --- | --- | --- |")
            for event in scenarios.worst_periods:
                lines.append(event.markdown_row())
            lines.append("")

    if confidence_intervals:
        lines.append("## Bootstrapped Confidence Intervals")
        lines.append("")
        lines.append("| Metric | 95% CI |")
        lines.append("| --- | --- |")
        for name in sorted(confidence_intervals.keys()):
            interval = confidence_intervals[name]
            lines.append(
                f"| {name.replace('_', ' ').title()} | {interval.format_markdown()} |"
            )
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
    milestones: Sequence[MilestoneReference] = (),
    profitability: ProfitabilitySummary | None = None,
    attribution: Sequence[AttributionRow] = (),
    regime_attribution: Sequence[RegimeAttributionGroup] = (),
    scenarios: ScenarioAnalysis | None = None,
    confidence_intervals: Mapping[str, ConfidenceInterval] = (),
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

    milestone_list = (
        "<h2>Research Agenda Alignment</h2>"
        f"<ul>{''.join(ms.to_html() for ms in milestones)}</ul>"
        if milestones
        else ""
    )

    profitability_html = ""
    if profitability is not None:
        prof_rows = "".join(
            f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in profitability.rows()
        )
        profitability_html = (
            "<h2>Profitability Summary</h2>"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
            f"{prof_rows}</tbody></table>"
        )

    attribution_html = ""
    if attribution:
        attr_rows = "".join(row.html_row() for row in attribution)
        attribution_html = (
            "<h2>Attribution by Symbol</h2>"
            "<table><thead><tr><th>Symbol</th><th>Sample Weight</th><th>Average Return</th>"
            "<th>Cumulative PnL</th><th>Contribution</th></tr></thead><tbody>"
            f"{attr_rows}</tbody></table>"
        )

    regime_html = ""
    if regime_attribution:
        sections: list[str] = ["<h2>Regime Attribution</h2>"]
        for group in regime_attribution:
            rows = "".join(row.html_row() for row in group.rows)
            display = group.name.replace("_", " ").title()
            sections.append(f"<h3>{display}</h3>")
            sections.append(
                "<table><thead><tr><th>Label</th><th>Samples</th><th>Sample Weight</th>"
                "<th>Average Return</th><th>Cumulative Return</th><th>Sharpe</th><th>Hit Rate</th>"
                "</tr></thead><tbody>"
                f"{rows}</tbody></table>"
            )
        regime_html = "".join(sections)

    scenario_html = ""
    if scenarios is not None and (
        scenarios.best_periods or scenarios.worst_periods or scenarios.drawdown
    ):
        parts: list[str] = ["<h2>Scenario Analysis</h2>"]
        if scenarios.drawdown is not None:
            parts.append(scenarios.drawdown.html_block())
        if scenarios.best_periods:
            best_rows = "".join(event.html_row() for event in scenarios.best_periods)
            parts.append("<h3>Best Periods</h3>")
            parts.append(
                "<table><thead><tr><th>Timestamp</th><th>Symbol</th><th>Return</th></tr></thead>"
                f"<tbody>{best_rows}</tbody></table>"
            )
        if scenarios.worst_periods:
            worst_rows = "".join(event.html_row() for event in scenarios.worst_periods)
            parts.append("<h3>Worst Periods</h3>")
            parts.append(
                "<table><thead><tr><th>Timestamp</th><th>Symbol</th><th>Return</th></tr></thead>"
                f"<tbody>{worst_rows}</tbody></table>"
            )
        scenario_html = "".join(parts)

    confidence_html = ""
    if confidence_intervals:
        ci_rows = "".join(
            f"<tr><td>{name.replace('_', ' ').title()}</td><td>{interval.format_html()}</td></tr>"
            for name, interval in sorted(confidence_intervals.items())
        )
        confidence_html = (
            "<h2>Bootstrapped Confidence Intervals</h2>"
            "<table><thead><tr><th>Metric</th><th>95% CI</th></tr></thead><tbody>"
            f"{ci_rows}</tbody></table>"
        )

    body = "\n".join(
        [
            f"<h1>{title}</h1>",
            description_html,
            "<h2>Overview</h2>",
            f"<ul>{''.join(overview_items)}</ul>",
            "<h2>Risk Metrics</h2>",
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>",
            f"{metrics_rows}</tbody></table>",
            profitability_html,
            milestone_list,
            attribution_html,
            regime_html,
            scenario_html,
            confidence_html,
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
    benchmark_column: str | None = None,
    title: str = "Market NN Plus Ultra Performance Report",
    description: str | None = None,
    include_equity_chart: bool = True,
    include_distribution_chart: bool = True,
    periods_per_year: int = 252,
    charts_dir_name: str | None = None,
    milestones: Optional[Iterable[MilestoneReference | Mapping[str, str]]] = None,
) -> Path:
    """Write a Markdown performance report and return the file path."""

    output = Path(output_path)
    if not output.suffix:
        output = output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_report(
        predictions,
        return_column=return_column,
        benchmark_column=benchmark_column,
        metrics=metrics,
        periods_per_year=periods_per_year,
    )

    charts_dir = output.parent / (
        charts_dir_name if charts_dir_name else f"{output.stem}_assets"
    )
    charts = _generate_charts(
        prepared.returns,
        charts_dir,
        include_equity=include_equity_chart,
        include_distribution=include_distribution_chart,
        periods_per_year=periods_per_year,
    )

    markdown_text = _build_markdown(
        title=title,
        description=description,
        summary=prepared.summary,
        metrics=prepared.metrics,
        charts=charts,
        output_dir=output.parent,
        milestones=_normalise_milestones(milestones),
        profitability=prepared.profitability,
        attribution=prepared.attribution,
        regime_attribution=prepared.regime_attribution,
        scenarios=prepared.scenarios,
        confidence_intervals=prepared.confidence_intervals,
    )
    output.write_text(markdown_text, encoding="utf-8")
    return output


def generate_html_report(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    metrics: Optional[Mapping[str, float]] = None,
    return_column: str = "realised_return",
    benchmark_column: str | None = None,
    title: str = "Market NN Plus Ultra Performance Report",
    description: str | None = None,
    include_equity_chart: bool = True,
    include_distribution_chart: bool = True,
    periods_per_year: int = 252,
    charts_dir_name: str | None = None,
    milestones: Optional[Iterable[MilestoneReference | Mapping[str, str]]] = None,
) -> Path:
    """Write an HTML performance report and return the file path."""

    output = Path(output_path)
    if not output.suffix:
        output = output.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_report(
        predictions,
        return_column=return_column,
        benchmark_column=benchmark_column,
        metrics=metrics,
        periods_per_year=periods_per_year,
    )

    charts_dir = output.parent / (
        charts_dir_name if charts_dir_name else f"{output.stem}_assets"
    )
    charts = _generate_charts(
        prepared.returns,
        charts_dir,
        include_equity=include_equity_chart,
        include_distribution=include_distribution_chart,
        periods_per_year=periods_per_year,
    )

    html_text = _build_html(
        title=title,
        description=description,
        summary=prepared.summary,
        metrics=prepared.metrics,
        charts=charts,
        output_dir=output.parent,
        milestones=_normalise_milestones(milestones),
        profitability=prepared.profitability,
        attribution=prepared.attribution,
        regime_attribution=prepared.regime_attribution,
        scenarios=prepared.scenarios,
        confidence_intervals=prepared.confidence_intervals,
    )
    output.write_text(html_text, encoding="utf-8")
    return output


def generate_report(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    metrics: Optional[Mapping[str, float]] = None,
    return_column: str = "realised_return",
    benchmark_column: str | None = None,
    title: str = "Market NN Plus Ultra Performance Report",
    description: str | None = None,
    include_equity_chart: bool = True,
    include_distribution_chart: bool = True,
    periods_per_year: int = 252,
    charts_dir_name: str | None = None,
    milestones: Optional[Iterable[MilestoneReference | Mapping[str, str]]] = None,
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
            benchmark_column=benchmark_column,
            title=title,
            description=description,
            include_equity_chart=include_equity_chart,
            include_distribution_chart=include_distribution_chart,
            periods_per_year=periods_per_year,
            charts_dir_name=charts_dir_name,
            milestones=milestones,
        )
    return generate_markdown_report(
        predictions,
        output if suffix else output.with_suffix(".md"),
        metrics=metrics,
        return_column=return_column,
        benchmark_column=benchmark_column,
        title=title,
        description=description,
        include_equity_chart=include_equity_chart,
        include_distribution_chart=include_distribution_chart,
        periods_per_year=periods_per_year,
        charts_dir_name=charts_dir_name,
        milestones=milestones,
    )


__all__ = [
    "ReportSummary",
    "MilestoneReference",
    "generate_markdown_report",
    "generate_html_report",
    "generate_report",
]
