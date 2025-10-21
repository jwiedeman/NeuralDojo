from datetime import datetime

import pandas as pd
import pytest

from market_nn_plus_ultra.trading.guardrails import GuardrailConfig, GuardrailPolicy


def _base_trades() -> pd.DataFrame:
    timestamps = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 2),
    ]
    data = {
        "timestamp": timestamps,
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "sector": ["tech", "finance", "tech", "finance"],
        "notional": [200.0, 40.0, 200.0, 40.0],
        "price": [10.0, 10.0, 10.0, 10.0],
        "position": [20.0, 4.0, 20.0, 4.0],
        "pnl": [-10.0, 2.0, -8.0, 1.0],
    }
    return pd.DataFrame(data)


def test_guardrail_policy_scales_global_exposure() -> None:
    trades = _base_trades()
    policy = GuardrailPolicy(
        GuardrailConfig(
            enabled=True,
            capital_base=200.0,
            max_gross_exposure=0.5,
            max_net_exposure=1.0,
            max_symbol_exposure=1.0,
        )
    )

    result = policy.enforce(trades)

    assert result.scaled is True
    assert result.metrics["gross_exposure_peak"] == pytest.approx(0.5, rel=1e-6)
    scaled_ratio = result.trades.loc[0, "notional"] / trades.loc[0, "notional"]
    assert scaled_ratio < 1.0
    assert result.trades.loc[0, "position"] == pytest.approx(trades.loc[0, "position"] * scaled_ratio)


def test_guardrail_policy_respects_sector_caps() -> None:
    trades = _base_trades()
    policy = GuardrailPolicy(
        GuardrailConfig(
            enabled=True,
            capital_base=200.0,
            sector_caps={"tech": 0.3},
        )
    )

    result = policy.enforce(trades)

    tech_exposure = result.exposures["sector"]["tech"]
    assert tech_exposure == pytest.approx(0.3, rel=1e-6)
    finance_rows = result.trades[result.trades["sector"] == "finance"]
    assert finance_rows["notional"].iloc[0] == pytest.approx(trades.loc[1, "notional"])


def test_guardrail_policy_records_tail_violation_when_warn_only() -> None:
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "notional": [50.0, 50.0, 50.0],
            "price": [5.0, 5.0, 5.0],
            "position": [10.0, 10.0, 10.0],
            "pnl": [-0.1, -0.05, 0.02],
        }
    )
    policy = GuardrailPolicy(
        GuardrailConfig(
            enabled=True,
            capital_base=100.0,
            min_tail_return=-0.02,
            tail_percentile=5.0,
            enforcement="warn",
        )
    )

    result = policy.enforce(trades)

    assert result.scaled is False
    violation_names = {violation.name for violation in result.violations}
    assert "tail_return_quantile" in violation_names
    assert result.metrics["tail_return_quantile"] < -0.02


def test_guardrail_policy_exposes_symbol_and_sector_summaries() -> None:
    trades = _base_trades()
    policy = GuardrailPolicy(
        GuardrailConfig(
            enabled=False,
            capital_base=200.0,
        )
    )

    result = policy.enforce(trades)

    exposures = result.exposures
    assert "symbol" in exposures
    assert exposures["symbol"]["AAA"] == pytest.approx(1.0)
    assert exposures["symbol"]["BBB"] == pytest.approx(0.2)
    assert exposures.get("sector", {})["tech"] == pytest.approx(1.0)
