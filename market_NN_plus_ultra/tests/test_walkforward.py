import numpy as np
import pandas as pd

from market_nn_plus_ultra.evaluation.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    generate_walk_forward_splits,
)


def test_generate_walk_forward_splits_basic() -> None:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="D")
    splits = generate_walk_forward_splits(timestamps, train_window=6, test_window=3)
    assert len(splits) == 2
    assert splits[0].train_start == timestamps[0]
    assert splits[0].test_end == timestamps[8]
    assert splits[1].train_start == timestamps[3]


def test_walk_forward_backtester_runs_metrics() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D")
    returns = np.linspace(-0.01, 0.02, len(timestamps))
    frame = pd.DataFrame(
        {
            "window_end": timestamps,
            "realised_return": returns,
        }
    )
    config = WalkForwardConfig(train_window=5, test_window=3)
    backtester = WalkForwardBacktester(config)
    result = backtester.run(frame)
    assert not result.empty
    assert {"metric_sharpe", "metric_max_drawdown"}.issubset(result.columns)
