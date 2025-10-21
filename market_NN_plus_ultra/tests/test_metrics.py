"""Tests for risk metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from market_nn_plus_ultra.evaluation.metrics import risk_metrics


def test_risk_metrics_with_benchmark_produces_excess_statistics() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.015], dtype=np.float64)
    benchmark = np.array([0.008, 0.015, -0.005, 0.01], dtype=np.float64)

    metrics = risk_metrics(returns, benchmark_returns=benchmark, periods_per_year=252)

    assert "tracking_error" in metrics
    assert "information_ratio" in metrics
    assert "beta" in metrics
    assert "alpha" in metrics
    assert "benchmark_correlation" in metrics
    assert "average_excess_return" in metrics

    excess = returns - benchmark
    tracking = excess.std(ddof=0) * np.sqrt(252)
    expected_ir = (excess.mean() * 252) / tracking
    covariance = np.cov(returns, benchmark, ddof=0)[0, 1]
    beta = covariance / benchmark.var(ddof=0)
    alpha = (returns.mean() * 252) - beta * (benchmark.mean() * 252)
    correlation = np.corrcoef(returns, benchmark)[0, 1]

    assert metrics["tracking_error"] == pytest.approx(tracking)
    assert metrics["information_ratio"] == pytest.approx(expected_ir)
    assert metrics["beta"] == pytest.approx(beta)
    assert metrics["alpha"] == pytest.approx(alpha)
    assert metrics["benchmark_correlation"] == pytest.approx(correlation)
    assert metrics["average_excess_return"] == pytest.approx(excess.mean())
    assert metrics["annualised_excess_return"] == pytest.approx(excess.mean() * 252)
    assert metrics["benchmark_average_return"] == pytest.approx(benchmark.mean())
    assert metrics["benchmark_annualised_return"] == pytest.approx(benchmark.mean() * 252)
