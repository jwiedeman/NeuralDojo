import numpy as np
import pandas as pd
import pytest

from market_nn_plus_ultra.evaluation import guardrail_metrics


def _build_sample_trades() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D")
    symbols = ["AAA", "BBB"] * 3
    notional = np.array([100.0, -40.0, 150.0, -60.0, 80.0, -20.0])
    pnl = np.array([0.5, -0.2, 0.4, -0.1, -0.3, 0.2])
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "symbol": symbols,
            "notional": notional,
            "price": np.full_like(notional, 10.0),
            "position": notional / 10.0,
            "pnl": pnl,
        }
    )
    return df


def test_guardrail_metrics_from_notional() -> None:
    trades = _build_sample_trades()
    metrics = guardrail_metrics(trades, capital_base=200.0, tail_percentile=5.0)

    assert metrics["gross_exposure_peak"] == pytest.approx(1.05)
    assert metrics["gross_exposure_mean"] == pytest.approx(0.75)
    assert metrics["net_exposure_mean"] == pytest.approx(0.35)
    assert metrics["net_exposure_peak"] == pytest.approx(0.45)
    assert metrics["turnover_rate"] == pytest.approx(0.1)
    assert metrics["tail_return_quantile"] == pytest.approx(-0.06)
    assert metrics["tail_return_mean"] == pytest.approx(-0.1)
    assert metrics["tail_event_frequency"] == pytest.approx(1 / 3)
    assert metrics["max_symbol_exposure"] == pytest.approx(0.75)


def test_guardrail_metrics_derives_notional_from_position_price() -> None:
    trades = _build_sample_trades().drop(columns=["notional"])
    metrics = guardrail_metrics(trades, capital_base=200.0)

    baseline = guardrail_metrics(_build_sample_trades(), capital_base=200.0)
    for key, value in baseline.items():
        assert metrics[key] == pytest.approx(value)


def test_guardrail_metrics_requires_exposure_inputs() -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="D")
    trades = pd.DataFrame({"timestamp": timestamps, "position": [1.0, -1.0]})

    with pytest.raises(ValueError):
        guardrail_metrics(trades)
