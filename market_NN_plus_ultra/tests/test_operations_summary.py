import pandas as pd

from market_nn_plus_ultra.evaluation import (
    OperationsThresholds,
    compile_operations_summary,
)


def test_compile_operations_summary_without_trades_returns_risk_only() -> None:
    predictions = pd.DataFrame(
        {
            "window_end": pd.date_range("2025-01-01", periods=4, freq="D"),
            "realised_return": [0.01, -0.02, 0.015, 0.005],
        }
    )

    summary = compile_operations_summary(predictions)

    assert summary.guardrails is None
    assert summary.triggered == []
    payload = summary.as_dict()
    assert "risk" in payload
    assert "guardrails" not in payload
    assert payload["risk"]["sharpe"] == summary.risk["sharpe"]


def test_compile_operations_summary_triggers_thresholds() -> None:
    predictions = pd.DataFrame(
        {
            "window_end": pd.date_range("2025-01-01", periods=5, freq="D"),
            "realised_return": [0.02, -0.03, 0.015, -0.025, 0.01],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
            "symbol": ["AAA", "AAA", "BBB", "BBB", "BBB"],
            "position": [100, -80, 120, -90, 140],
            "price": [10.0, 10.5, 9.5, 9.8, 10.2],
            "pnl": [200.0, -260.0, 150.0, -220.0, 175.0],
        }
    )

    thresholds = OperationsThresholds(
        min_sharpe=1.5,
        max_drawdown=0.01,
        max_gross_exposure=0.4,
        max_turnover=0.05,
        min_tail_return=-150.0,
        max_tail_frequency=0.1,
        max_symbol_exposure=0.5,
    )

    summary = compile_operations_summary(
        predictions,
        trades,
        thresholds=thresholds,
        capital_base=1000.0,
    )

    assert summary.guardrails is not None
    assert summary.triggered
    assert any("Sharpe" in message for message in summary.triggered)
    assert any("Gross exposure" in message for message in summary.triggered)
    assert any("Tail return" in message for message in summary.triggered)
    assert any("Symbol exposure" in message for message in summary.triggered)
    payload = summary.as_dict()
    assert payload["guardrails"]["gross_exposure_peak"] == summary.guardrails["gross_exposure_peak"]
