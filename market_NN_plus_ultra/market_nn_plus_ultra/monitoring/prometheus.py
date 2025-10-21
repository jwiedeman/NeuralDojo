"""Prometheus exporters for live monitoring metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any

try:  # pragma: no cover - optional dependency guard
    from prometheus_client import CollectorRegistry, Counter, Gauge
except ModuleNotFoundError as exc:  # pragma: no cover - runtime fallback
    CollectorRegistry = Counter = Gauge = None  # type: ignore[assignment]
    _PROMETHEUS_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised via integration tests
    _PROMETHEUS_IMPORT_ERROR = None

from .live import MonitoringSnapshot


if _PROMETHEUS_IMPORT_ERROR is None:
    MONITORING_RISK_GAUGE = Gauge(
        "plus_ultra_monitoring_risk_metric",
        "Risk metric values computed by the live monitor.",
        labelnames=("name",),
    )

    MONITORING_DRIFT_GAUGE = Gauge(
        "plus_ultra_monitoring_drift_metric",
        "Distribution drift metrics produced by the live monitor.",
        labelnames=("name",),
    )

    MONITORING_WINDOW_GAUGE = Gauge(
        "plus_ultra_monitoring_window_size",
        "Current number of returns stored in the monitoring window.",
    )

    MONITORING_ALERT_COUNTER = Counter(
        "plus_ultra_monitoring_alerts_total",
        "Total number of alerts emitted by the live monitor.",
        labelnames=("alert",),
    )

    LAST_ALERT_TIMESTAMP = Gauge(
        "plus_ultra_monitoring_last_alert_timestamp",
        "Unix timestamp of the most recent alert emission.",
    )
else:  # pragma: no cover - executed when prometheus-client is unavailable
    MONITORING_RISK_GAUGE = None
    MONITORING_DRIFT_GAUGE = None
    MONITORING_WINDOW_GAUGE = None
    MONITORING_ALERT_COUNTER = None
    LAST_ALERT_TIMESTAMP = None


@dataclass(slots=True)
class PrometheusExporter:
    """Push monitoring snapshots into Prometheus metrics."""

    risk_gauge: Any
    drift_gauge: Any
    window_gauge: Any
    alert_counter: Any
    last_alert_gauge: Any

    def publish(self, snapshot: MonitoringSnapshot) -> None:
        """Update gauges and counters to reflect *snapshot*."""

        if _PROMETHEUS_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "prometheus_client is required to publish monitoring metrics"
            ) from _PROMETHEUS_IMPORT_ERROR

        for name, value in snapshot.risk.items():
            self.risk_gauge.labels(name=name).set(float(value))

        for name, value in snapshot.drift.as_dict().items():
            self.drift_gauge.labels(name=name).set(float(value))

        self.window_gauge.set(float(snapshot.window_count))

        if snapshot.alerts:
            timestamp = time()
            for alert in snapshot.alerts:
                self.alert_counter.labels(alert=alert).inc()
            self.last_alert_gauge.set(timestamp)


def default_prometheus_exporter() -> PrometheusExporter:
    """Return an exporter wired to module-level metrics."""

    if _PROMETHEUS_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "prometheus_client is required to construct a Prometheus exporter"
        ) from _PROMETHEUS_IMPORT_ERROR

    return PrometheusExporter(
        risk_gauge=MONITORING_RISK_GAUGE,
        drift_gauge=MONITORING_DRIFT_GAUGE,
        window_gauge=MONITORING_WINDOW_GAUGE,
        alert_counter=MONITORING_ALERT_COUNTER,
        last_alert_gauge=LAST_ALERT_TIMESTAMP,
    )


def make_prometheus_exporter(registry: CollectorRegistry) -> PrometheusExporter:
    """Return an exporter using gauges registered on *registry*."""

    if _PROMETHEUS_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "prometheus_client is required to construct a Prometheus exporter"
        ) from _PROMETHEUS_IMPORT_ERROR

    risk = Gauge(
        "plus_ultra_monitoring_risk_metric",
        "Risk metric values computed by the live monitor.",
        labelnames=("name",),
        registry=registry,
    )
    drift = Gauge(
        "plus_ultra_monitoring_drift_metric",
        "Distribution drift metrics produced by the live monitor.",
        labelnames=("name",),
        registry=registry,
    )
    window = Gauge(
        "plus_ultra_monitoring_window_size",
        "Current number of returns stored in the monitoring window.",
        registry=registry,
    )
    alerts = Counter(
        "plus_ultra_monitoring_alerts_total",
        "Total number of alerts emitted by the live monitor.",
        labelnames=("alert",),
        registry=registry,
    )
    last_alert = Gauge(
        "plus_ultra_monitoring_last_alert_timestamp",
        "Unix timestamp of the most recent alert emission.",
        registry=registry,
    )
    return PrometheusExporter(risk, drift, window, alerts, last_alert)


__all__ = [
    "PrometheusExporter",
    "default_prometheus_exporter",
    "make_prometheus_exporter",
]
