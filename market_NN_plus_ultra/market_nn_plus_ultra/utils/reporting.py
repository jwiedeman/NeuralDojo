"""Reporting helpers for presenting and persisting evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping

import json
import math


def _format_value(value: object, precision: int) -> str:
    """Return a human-friendly representation for a metric value."""

    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        format_spec = f".{{precision}}f".format(precision=precision)
        return format(value, format_spec)
    if value is None:
        return "-"
    return str(value)


def sanitize_metrics(metrics: Mapping[str, object], *, nan_fill: float = 0.0) -> dict[str, float | object]:
    """Return a copy of *metrics* with NaNs replaced for downstream consumers."""

    cleaned: MutableMapping[str, float | object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if math.isnan(value):
                cleaned[key] = nan_fill
            else:
                cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return dict(cleaned)


def format_metrics_table(metrics: Mapping[str, object], *, precision: int = 6) -> str:
    """Return a plain-text table formatted from the provided *metrics*."""

    if not metrics:
        return "(no metrics)"

    rows = sorted(metrics.items(), key=lambda item: item[0])
    name_width = max(len("Metric"), *(len(name) for name, _ in rows))
    value_width = max(
        len("Value"),
        *(
            len(_format_value(value, precision))
            for _, value in rows
        ),
    )

    header = f"{'Metric'.ljust(name_width)} | {'Value'.ljust(value_width)}"
    divider = f"{'-' * name_width}-+-{'-' * value_width}"
    body = [
        f"{name.ljust(name_width)} | {_format_value(value, precision).rjust(value_width)}"
        for name, value in rows
    ]
    return "\n".join([header, divider, *body])


def _serialisable_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """Convert metrics to JSON/CSV-friendly values."""

    serialisable: MutableMapping[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if math.isnan(value):
                serialisable[key] = None
            elif math.isinf(value):
                serialisable[key] = "inf" if value > 0 else "-inf"
            else:
                serialisable[key] = float(value)
        else:
            serialisable[key] = value
    return dict(serialisable)


def write_metrics_report(
    metrics: Mapping[str, object],
    path: Path,
    *,
    precision: int = 6,
    format_hint: str | None = None,
) -> Path:
    """Persist *metrics* to *path* in a human-friendly format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = _serialisable_metrics(metrics)

    fmt = (format_hint or path.suffix.lstrip(".").lower()) or "txt"
    if fmt == "json":
        path.write_text(json.dumps(serialisable, indent=2, sort_keys=True), encoding="utf-8")
    elif fmt in {"csv", "tsv"}:
        delimiter = "," if fmt == "csv" else "\t"
        lines = ["metric" + delimiter + "value"]
        for name in sorted(serialisable):
            value = serialisable[name]
            lines.append(f"{name}{delimiter}{value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif fmt in {"md", "markdown"}:
        lines = ["| Metric | Value |", "| --- | --- |"]
        for name in sorted(metrics):
            value = _format_value(metrics[name], precision)
            lines.append(f"| {name} | {value} |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        table = format_metrics_table(metrics, precision=precision)
        path.write_text(table + "\n", encoding="utf-8")

    return path


__all__ = ["format_metrics_table", "sanitize_metrics", "write_metrics_report"]
