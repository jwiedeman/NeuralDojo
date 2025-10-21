"""Live monitoring helpers for Plus Ultra deployments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Sequence

import numpy as np

from ..evaluation.metrics import risk_metrics
from ..evaluation.operations import OperationsThresholds
from .drift import DriftMetrics, compute_drift_metrics


@dataclass(slots=True)
class DriftAlertThresholds:
    """Thresholds that trigger drift alerts when exceeded."""

    psi_alert: float = 0.2
    js_alert: float = 0.05
    ks_alert: float = 0.15


@dataclass(slots=True)
class MonitoringSnapshot:
    """Summary of the latest monitoring window."""

    risk: dict[str, float]
    drift: DriftMetrics
    alerts: list[str]
    window_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable representation of the snapshot."""

        payload: dict[str, object] = {
            "risk": dict(self.risk),
            "drift": self.drift.as_dict(),
            "alerts": list(self.alerts),
            "window_count": int(self.window_count),
        }
        return payload


class LiveMonitor:
    """Rolling monitor that tracks performance and distribution drift."""

    __slots__ = (
        "_reference",
        "_window",
        "_window_size",
        "_drift_bins",
        "_risk_thresholds",
        "_drift_thresholds",
    )

    def __init__(
        self,
        reference_returns: Sequence[float] | np.ndarray,
        *,
        window_size: int = 512,
        drift_bins: int = 20,
        risk_thresholds: OperationsThresholds | None = None,
        drift_thresholds: DriftAlertThresholds | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if drift_bins <= 0:
            raise ValueError("drift_bins must be positive")

        reference = np.asarray(reference_returns, dtype=np.float64)
        reference = reference[np.isfinite(reference)]
        if reference.size == 0:
            raise ValueError("reference_returns must contain at least one finite value")

        self._reference = reference
        self._window: Deque[float] = deque(maxlen=window_size)
        self._window_size = window_size
        self._drift_bins = drift_bins
        self._risk_thresholds = risk_thresholds or OperationsThresholds()
        self._drift_thresholds = drift_thresholds or DriftAlertThresholds()

    @property
    def reference_returns(self) -> np.ndarray:
        """Return a copy of the reference returns."""

        return self._reference.copy()

    @property
    def window_size(self) -> int:
        return self._window_size

    def reset(self) -> None:
        """Clear the monitoring window."""

        self._window.clear()

    def update(self, returns: Iterable[float]) -> MonitoringSnapshot:
        """Consume *returns* and emit an updated snapshot."""

        for value in returns:
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                self._window.append(value)

        window_array = np.asarray(self._window, dtype=np.float64)

        metrics = risk_metrics(window_array)
        drift = compute_drift_metrics(self._reference, window_array, bins=self._drift_bins)

        alerts: list[str] = []
        thresholds = self._risk_thresholds

        sharpe = float(metrics.get("sharpe", 0.0))
        if thresholds.min_sharpe is not None and sharpe < thresholds.min_sharpe:
            alerts.append(
                f"Sharpe ratio {sharpe:.3f} below threshold {thresholds.min_sharpe:.3f}"
            )

        drawdown = float(abs(metrics.get("max_drawdown", 0.0)))
        if thresholds.max_drawdown is not None and drawdown > thresholds.max_drawdown:
            alerts.append(
                f"Max drawdown {drawdown:.3f} exceeded limit {thresholds.max_drawdown:.3f}"
            )

        tail_return = float(metrics.get("value_at_risk", 0.0))
        if thresholds.min_tail_return is not None and tail_return < thresholds.min_tail_return:
            alerts.append(
                f"Value-at-Risk {tail_return:.3f} breached floor {thresholds.min_tail_return:.3f}"
            )

        drift_thresholds = self._drift_thresholds
        if drift.population_stability_index > drift_thresholds.psi_alert:
            alerts.append(
                "Population stability index "
                f"{drift.population_stability_index:.3f} exceeded {drift_thresholds.psi_alert:.3f}"
            )
        if drift.js_divergence > drift_thresholds.js_alert:
            alerts.append(
                "Jensen-Shannon divergence "
                f"{drift.js_divergence:.3f} exceeded {drift_thresholds.js_alert:.3f}"
            )
        if drift.ks_statistic > drift_thresholds.ks_alert:
            alerts.append(
                "KS statistic "
                f"{drift.ks_statistic:.3f} exceeded {drift_thresholds.ks_alert:.3f}"
            )

        snapshot = MonitoringSnapshot(
            risk=metrics,
            drift=drift,
            alerts=alerts,
            window_count=window_array.size,
        )
        return snapshot

    def window_returns(self) -> np.ndarray:
        """Return a copy of the current monitoring window."""

        return np.asarray(self._window, dtype=np.float64)


__all__ = [
    "DriftAlertThresholds",
    "LiveMonitor",
    "MonitoringSnapshot",
]
