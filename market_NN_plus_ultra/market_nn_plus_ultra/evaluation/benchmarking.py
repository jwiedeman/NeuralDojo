"""Utilities for analysing architecture sweep benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(slots=True)
class ArchitectureSummary:
    """Aggregated metrics describing an architecture's benchmark performance."""

    architecture: str
    scenario_count: int
    best_metric: float | None
    best_metric_label: str | None
    best_model_path: str | None
    median_metric: float | None
    mean_metric: float | None
    median_duration: float | None
    mean_duration: float | None
    mean_profitability: float | None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "architecture": self.architecture,
            "scenario_count": self.scenario_count,
            "best_metric": self.best_metric,
            "best_metric_label": self.best_metric_label,
            "best_model_path": self.best_model_path,
            "median_metric": self.median_metric,
            "mean_metric": self.mean_metric,
            "median_duration": self.median_duration,
            "mean_duration": self.mean_duration,
            "mean_profitability": self.mean_profitability,
        }


def _ensure_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Required column '{column}' not present in benchmark frame")
    return frame[column]


def _coerce_float(value: object | pd.Series | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        numeric = pd.to_numeric(value, errors="coerce").dropna()
        if numeric.empty:
            return None
        result = float(numeric.mean())
        return None if math.isnan(result) else result
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None
    return None if math.isnan(result) else result


def _to_numeric_series(series: object | pd.Series | None) -> pd.Series | None:
    if not isinstance(series, pd.Series):
        return None if series is None else pd.Series([series], dtype="float64")
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric


def summarise_architecture_performance(
    frame: pd.DataFrame,
    *,
    metric: str = "metric_val_loss",
    higher_is_better: bool = False,
    profitability_metric: str = "profitability_roi",
) -> list[ArchitectureSummary]:
    """Summarise benchmark results per architecture.

    Parameters
    ----------
    frame:
        Benchmark rows emitted by :mod:`scripts/benchmarks/architecture_sweep.py`.
    metric:
        Column name used to rank architectures (defaults to ``metric_val_loss``).
    higher_is_better:
        Whether larger metric values indicate better performance. Loss metrics
        should keep the default ``False`` while profitability metrics may want
        ``True``.
    profitability_metric:
        Optional column to average for profitability comparisons. Missing
        columns are ignored gracefully.
    """

    if frame.empty:
        raise ValueError("Benchmark frame is empty; nothing to summarise")

    _ensure_column(frame, metric)
    ascending = not higher_is_better

    summaries: list[ArchitectureSummary] = []
    for architecture, group in frame.groupby("architecture", sort=False):
        clean_group = group.dropna(subset=[metric]).copy()
        clean_group[metric] = pd.to_numeric(clean_group[metric], errors="coerce")
        clean_group = clean_group.dropna(subset=[metric])
        if clean_group.empty:
            summaries.append(
                ArchitectureSummary(
                    architecture=str(architecture),
                    scenario_count=int(group.shape[0]),
                    best_metric=None,
                    best_metric_label=None,
                    best_model_path=None,
                    median_metric=None,
                    mean_metric=None,
                    median_duration=_coerce_float(group.get("duration_seconds")),
                    mean_duration=_coerce_float(group.get("duration_seconds")),
                    mean_profitability=_coerce_float(group.get(profitability_metric)),
                )
            )
            continue

        clean_group[metric] = clean_group[metric].astype(float)
        sorted_group = clean_group.sort_values(metric, ascending=ascending)
        best_row = sorted_group.iloc[0]

        best_metric = float(best_row[metric])
        best_label_obj = best_row.get("label") or best_row.get("scenario")
        best_label = str(best_label_obj) if best_label_obj is not None else None
        best_model_path_obj = best_row.get("best_model_path")
        best_model_path = str(best_model_path_obj) if isinstance(best_model_path_obj, str) else None

        duration_series = _to_numeric_series(clean_group.get("duration_seconds"))
        profitability_series = _to_numeric_series(clean_group.get(profitability_metric))

        summaries.append(
            ArchitectureSummary(
                architecture=str(architecture),
                scenario_count=int(group.shape[0]),
                best_metric=best_metric,
                best_metric_label=best_label,
                best_model_path=best_model_path,
                median_metric=_coerce_float(clean_group[metric].median()),
                mean_metric=_coerce_float(clean_group[metric].mean()),
                median_duration=_coerce_float(duration_series.median() if isinstance(duration_series, pd.Series) else duration_series),
                mean_duration=_coerce_float(duration_series.mean() if isinstance(duration_series, pd.Series) else duration_series),
                mean_profitability=_coerce_float(profitability_series.mean() if isinstance(profitability_series, pd.Series) else profitability_series),
            )
        )

    key = (lambda summary: float("-inf") if summary.best_metric is None else summary.best_metric)
    summaries.sort(key=key, reverse=higher_is_better)
    return summaries


def load_benchmark_frames(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Load and concatenate benchmark outputs from the provided paths."""

    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix == ".json":
            frame = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported benchmark format for '{path}'")
        frame = frame.copy()
        frame["__source"] = path.name
        frames.append(frame)

    if not frames:
        raise ValueError("No benchmark files provided")

    return pd.concat(frames, ignore_index=True, copy=False)


def summaries_to_frame(summaries: Iterable[ArchitectureSummary]) -> pd.DataFrame:
    """Convert summaries into a pandas DataFrame."""

    return pd.DataFrame([summary.to_dict() for summary in summaries])


def format_markdown_table(
    summaries: Sequence[ArchitectureSummary],
    *,
    metric: str,
) -> str:
    """Render a Markdown table for the provided summaries."""

    header = (
        "| Architecture | Runs | Best {metric} | Best Label | Median {metric} | Mean Duration (s) |"
        .format(metric=metric)
    )
    separator = "| --- | ---: | ---: | --- | ---: | ---: |"
    rows = [header, separator]
    for summary in summaries:
        best_metric = "—" if summary.best_metric is None else f"{summary.best_metric:.6f}"
        median_metric = "—" if summary.median_metric is None else f"{summary.median_metric:.6f}"
        mean_duration = "—" if summary.mean_duration is None else f"{summary.mean_duration:.2f}"
        rows.append(
            "| {arch} | {count} | {best} | {label} | {median} | {duration} |".format(
                arch=summary.architecture,
                count=summary.scenario_count,
                best=best_metric,
                label=summary.best_metric_label or "—",
                median=median_metric,
                duration=mean_duration,
            )
        )
    return "\n".join(rows)


def _normalise_group_by(group_by: Sequence[str] | None) -> list[str]:
    if group_by is None:
        return []
    return [column for column in group_by if column]


def _format_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "—"
    formatted = f"{value:.{precision}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def architecture_leaderboard(
    frame: pd.DataFrame,
    *,
    metric: str = "metric_val_loss",
    higher_is_better: bool = False,
    group_by: Sequence[str] | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Return the top-performing scenarios per group for rapid comparisons.

    This helper collapses raw benchmark rows into a leaderboard that highlights
    the strongest performing architectures per asset universe or fixture group.
    It is especially useful when triaging GPU runs for the omni-scale backbone
    and hybrid baselines: instead of manually inspecting large parquet files,
    the leaderboard surfaces the best-performing runs per configuration bucket
    using the requested metric.

    Parameters
    ----------
    frame:
        Benchmark rows emitted by :mod:`scripts/benchmarks/architecture_sweep.py`.
    metric:
        Column name used to rank rows (defaults to ``metric_val_loss``).
    higher_is_better:
        Whether larger metric values indicate better performance.
    group_by:
        Optional sequence of column names used to build independent leaderboards.
        When omitted the entire frame is considered one group.
    top_k:
        Number of rows to include per group. Must be positive.
    """

    if frame.empty:
        raise ValueError("Benchmark frame is empty; nothing to rank")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    _ensure_column(frame, metric)
    group_columns = _normalise_group_by(group_by)
    for column in group_columns:
        _ensure_column(frame, column)

    ascending = not higher_is_better
    groups: Iterable[tuple[tuple[object, ...], pd.DataFrame]]
    if group_columns:
        groups = (
            (key if isinstance(key, tuple) else (key,), group.copy())
            for key, group in frame.groupby(group_columns, sort=False, dropna=False)
        )
    else:
        groups = [(("all",), frame.copy())]

    rows: list[dict[str, object]] = []
    for key, group in groups:
        group = group.copy()
        group[metric] = pd.to_numeric(group[metric], errors="coerce")
        group = group.dropna(subset=[metric])
        if group.empty:
            continue

        sorted_group = group.sort_values(metric, ascending=ascending)
        limited = sorted_group.head(top_k)
        for rank, (_, row) in enumerate(limited.iterrows(), start=1):
            record: dict[str, object] = {
                "rank": rank,
                "architecture": row.get("architecture"),
                "label": row.get("label") or row.get("scenario"),
                metric: float(row[metric]),
                "duration_seconds": _coerce_float(row.get("duration_seconds")),
                "best_model_path": row.get("best_model_path"),
            }
            for column, value in zip(group_columns or ["group"], key):
                record[column] = value
            rows.append(record)

    if not rows:
        raise ValueError("No benchmark rows contained finite metric values")

    leaderboard_frame = pd.DataFrame(rows)
    column_order = list(group_columns or ["group"]) + [
        "rank",
        "architecture",
        "label",
        metric,
        "duration_seconds",
        "best_model_path",
    ]
    existing_columns = [column for column in column_order if column in leaderboard_frame.columns]
    leaderboard_frame = leaderboard_frame[existing_columns]
    return leaderboard_frame


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a :class:`~pandas.DataFrame` as a Markdown table."""

    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float) and not math.isnan(value):
                cells.append(_format_float(value))
            elif pd.isna(value):
                cells.append("—")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


__all__ = [
    "ArchitectureSummary",
    "architecture_leaderboard",
    "summarise_architecture_performance",
    "load_benchmark_frames",
    "summaries_to_frame",
    "format_markdown_table",
    "dataframe_to_markdown",
]
