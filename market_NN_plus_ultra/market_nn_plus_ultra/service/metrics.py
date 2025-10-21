"""Prometheus metrics helpers for the Market NN Plus Ultra service."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from time import perf_counter, time
from typing import Iterable, Optional

if find_spec("prometheus_client") is not None:  # pragma: no cover - exercised via integration tests
    from prometheus_client import (  # type: ignore[import-not-found]
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
else:  # pragma: no cover - covered by unit tests when dependency is absent
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _MetricRegistry:
        """Minimal in-memory registry for metrics when Prometheus is unavailable."""

        def __init__(self) -> None:
            self._metrics: list[_SimpleMetricBase] = []

        def register(self, metric: "_SimpleMetricBase") -> None:
            self._metrics.append(metric)

        def render(self) -> bytes:
            lines: list[str] = []
            for metric in self._metrics:
                lines.extend(metric.render())
            return ("\n".join(lines) + "\n").encode("utf-8")

    @dataclass
    class _Sample:
        value: float = 0.0

    class _MetricHandle:
        def __init__(self, metric: "_SimpleMetricBase", key: tuple[str, ...]) -> None:
            self._metric = metric
            self._key = key

        def inc(self, amount: float = 1.0) -> "_MetricHandle":
            self._metric._inc(self._key, amount)
            return self

        def observe(self, value: float) -> "_MetricHandle":
            self._metric._observe(self._key, value)
            return self

        def set(self, value: float) -> "_MetricHandle":
            self._metric._set(self._key, value)
            return self

    class _SimpleMetricBase:
        metric_type: str = "gauge"

        def __init__(
            self,
            name: str,
            documentation: str,
            *,
            labelnames: tuple[str, ...] = (),
            **_kwargs: object,
        ) -> None:
            self._name = name
            self._documentation = documentation
            self._labelnames = labelnames
            self._samples: dict[tuple[str, ...], _Sample] = {}
            _REGISTRY.register(self)

        def labels(self, **labels: str) -> _MetricHandle:
            key = tuple(labels.get(label, "") for label in self._labelnames)
            if key not in self._samples:
                self._samples[key] = _Sample()
            return _MetricHandle(self, key)

        # Direct operations for unlabeled metrics
        def inc(self, amount: float = 1.0) -> None:
            self._inc((), amount)

        def set(self, value: float) -> None:
            self._set((), value)

        def observe(self, value: float) -> None:
            self._observe((), value)

        def _inc(self, key: tuple[str, ...], amount: float) -> None:
            sample = self._samples.setdefault(key, _Sample())
            sample.value += float(amount)

        def _observe(self, key: tuple[str, ...], value: float) -> None:
            # Default behaviour matches increment semantics.
            self._inc(key, value)

        def _set(self, key: tuple[str, ...], value: float) -> None:
            sample = self._samples.setdefault(key, _Sample())
            sample.value = float(value)

        def render(self) -> list[str]:
            lines = [
                f"# HELP {self._name} {self._documentation}",
                f"# TYPE {self._name} {self.metric_type}",
            ]
            if not self._samples:
                value = 0.0
                sample_line = f"{self._name} {value}"
                lines.append(sample_line)
                return lines

            for key, sample in self._samples.items():
                if self._labelnames:
                    label_parts = [f'{label}="{val}"' for label, val in zip(self._labelnames, key)]
                    label_str = "{" + ",".join(label_parts) + "}"
                else:
                    label_str = ""
                lines.append(f"{self._name}{label_str} {sample.value}")
            return lines

    class Counter(_SimpleMetricBase):
        metric_type = "counter"

        def _observe(self, key: tuple[str, ...], value: float) -> None:
            # Histograms call observe, but counters should increase by the observed amount
            self._inc(key, value)

    class Gauge(_SimpleMetricBase):
        metric_type = "gauge"

    class Histogram(_SimpleMetricBase):
        metric_type = "histogram"

        def _observe(self, key: tuple[str, ...], value: float) -> None:
            # Track count of observations, ignoring bucket semantics.
            self._inc(key, 1.0)
            # Store the latest value to aid debugging.
            self._set(key + ("_last",), value)

        def render(self) -> list[str]:
            lines = [
                f"# HELP {self._name} {self._documentation}",
                f"# TYPE {self._name} {self.metric_type}",
            ]
            observed = False
            for key, sample in self._samples.items():
                if key and key[-1] == "_last":
                    labelnames = self._labelnames + ("stat",)
                    base_key = key[:-1]
                    label_parts = [f'{label}="{val}"' for label, val in zip(labelnames, base_key + ("last",))]
                    lines.append(f"{self._name}_bucket{{{','.join(label_parts)}}} {sample.value}")
                else:
                    label_parts = [f'{label}="{val}"' for label, val in zip(self._labelnames, key)]
                    lines.append(f"{self._name}_count{{{','.join(label_parts)}}} {sample.value}")
                    observed = True
            if not observed:
                lines.append(f"{self._name}_count 0")
            return lines

    _REGISTRY = _MetricRegistry()

    def generate_latest() -> bytes:
        return _REGISTRY.render()

REQUEST_COUNTER = Counter(
    "plus_ultra_service_requests_total",
    "Total number of HTTP requests handled by the inference service.",
    labelnames=("endpoint", "method", "status"),
)

REQUEST_LATENCY = Histogram(
    "plus_ultra_service_request_latency_seconds",
    "Latency distribution for inference service HTTP requests.",
    labelnames=("endpoint", "method"),
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

REQUEST_EXCEPTIONS = Counter(
    "plus_ultra_service_request_exceptions_total",
    "Total number of exceptions raised by endpoint handlers.",
    labelnames=("endpoint", "method", "exception"),
)

LAST_SUCCESS_TIMESTAMP = Gauge(
    "plus_ultra_service_last_success_timestamp",
    "Unix timestamp of the most recent successful request per endpoint.",
    labelnames=("endpoint",),
)

LAST_ERROR_TIMESTAMP = Gauge(
    "plus_ultra_service_last_error_timestamp",
    "Unix timestamp of the most recent failed request per endpoint.",
    labelnames=("endpoint",),
)

PREDICTION_ROWS = Histogram(
    "plus_ultra_service_prediction_rows",
    "Distribution of prediction row counts returned by the agent.",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000),
)

LAST_PREDICTION_ROWS = Gauge(
    "plus_ultra_service_last_prediction_rows",
    "Row count returned by the most recent prediction response.",
)

GUARDRAILS_ENABLED = Gauge(
    "plus_ultra_guardrails_enabled",
    "Whether guardrails are currently enabled for the service (1 enabled, 0 disabled).",
    labelnames=("policy",),
)

GUARDRAIL_VIOLATIONS = Counter(
    "plus_ultra_guardrail_violations_total",
    "Total number of guardrail violations detected by type.",
    labelnames=("name",),
)

LAST_GUARDRAIL_VIOLATIONS = Gauge(
    "plus_ultra_guardrail_last_violation_count",
    "Number of violations returned by the most recent guardrail evaluation.",
)


def observe_request(
    endpoint: str,
    method: str,
    status: int,
    duration_seconds: float,
    *,
    exception: Optional[str] = None,
) -> None:
    """Record metrics for an HTTP request."""

    REQUEST_COUNTER.labels(endpoint=endpoint, method=method, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration_seconds)

    if 200 <= status < 400:
        LAST_SUCCESS_TIMESTAMP.labels(endpoint=endpoint).set(time())
    else:
        LAST_ERROR_TIMESTAMP.labels(endpoint=endpoint).set(time())

    if exception is not None:
        REQUEST_EXCEPTIONS.labels(
            endpoint=endpoint,
            method=method,
            exception=exception,
        ).inc()


def observe_prediction_rows(row_count: int) -> None:
    """Track the distribution of prediction row counts."""

    if row_count < 0:
        return
    PREDICTION_ROWS.observe(row_count)
    LAST_PREDICTION_ROWS.set(row_count)


def record_guardrail_violations(violation_names: Iterable[str]) -> None:
    """Increment counters for guardrail violations emitted by the policy."""

    total = 0
    for name in violation_names:
        GUARDRAIL_VIOLATIONS.labels(name=name).inc()
        total += 1
    LAST_GUARDRAIL_VIOLATIONS.set(total)


def set_guardrail_enabled(enabled: bool, policy_label: str = "service") -> None:
    """Update the guardrail enabled gauge."""

    GUARDRAILS_ENABLED.labels(policy=policy_label).set(1.0 if enabled else 0.0)


def render_prometheus_metrics() -> tuple[bytes, str]:
    """Return the serialized Prometheus metrics payload and content type."""

    payload = generate_latest()
    return payload, CONTENT_TYPE_LATEST


class RequestTimer:
    """Context manager to time FastAPI endpoint handlers."""

    __slots__ = ("endpoint", "method", "start", "status", "exception")

    def __init__(self, endpoint: str, method: str) -> None:
        self.endpoint = endpoint
        self.method = method
        self.start = perf_counter()
        self.status = 200
        self.exception: Optional[str] = None

    def set_status(self, status: int) -> None:
        self.status = status

    def set_exception(self, exc: BaseException) -> None:
        self.exception = exc.__class__.__name__

    def __enter__(self) -> "RequestTimer":
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc_type is not None:
            if issubclass(exc_type, Exception):
                self.status = getattr(exc, "status_code", 500)  # type: ignore[attr-defined]
                self.exception = exc_type.__name__
        duration = perf_counter() - self.start
        observe_request(
            self.endpoint,
            self.method,
            self.status,
            duration,
            exception=self.exception,
        )


__all__ = [
    "RequestTimer",
    "observe_prediction_rows",
    "record_guardrail_violations",
    "render_prometheus_metrics",
    "set_guardrail_enabled",
]
