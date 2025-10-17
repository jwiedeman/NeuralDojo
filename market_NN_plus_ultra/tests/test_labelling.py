import numpy as np
import pandas as pd
import pandas.testing as pdt

from market_nn_plus_ultra.data.labelling import generate_regime_labels
from market_nn_plus_ultra.data.validation import validate_regime_frame


def _synthetic_price_panel(rows: int = 192) -> pd.DataFrame:
    timeline = pd.date_range("2020-01-01", periods=rows, freq="H")
    symbols = ["ALPHA", "BETA", "GAMMA"]
    phase_boundaries = np.array_split(np.arange(rows), 3)

    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols):
        rng = np.random.default_rng(idx + 1)
        drift_profile = [0.002, -0.001, 0.0015]
        drift_profile = np.roll(drift_profile, idx)
        vol_profile = [0.6, 1.8, 0.9]
        vol_profile = np.roll(vol_profile, idx)
        liquidity_profile = [0.5, 1.7, 1.0]
        liquidity_profile = np.roll(liquidity_profile, idx)

        returns_segments = []
        volume_segments = []
        for phase_idx, indices in enumerate(phase_boundaries):
            phase_len = len(indices)
            returns_segments.append(
                rng.normal(
                    loc=drift_profile[phase_idx],
                    scale=0.01 * vol_profile[phase_idx],
                    size=phase_len,
                )
            )
            base_volume = (idx + 1) * 2.5e5
            volume_segments.append(
                base_volume
                * liquidity_profile[phase_idx]
                * (1 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, phase_len)))
            )

        returns = np.concatenate(returns_segments)
        volume = np.concatenate(volume_segments)
        close = 100 * np.exp(np.cumsum(returns))
        open_price = close * (1 + rng.normal(0.0, 0.002, size=rows))
        high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.001, size=rows)))
        low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.001, size=rows)))
        turnover = close * volume

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timeline,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "turnover": turnover,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def test_generate_regime_labels_produces_expected_categories() -> None:
    price_df = _synthetic_price_panel()
    regimes = generate_regime_labels(price_df)

    assert set(regimes["name"].unique()) == {
        "volatility_regime",
        "liquidity_regime",
        "rotation_role",
    }
    assert not regimes.duplicated(["timestamp", "symbol", "name"]).any()

    vol_values = set(regimes.loc[regimes["name"] == "volatility_regime", "value"].unique())
    assert {"low_vol", "mid_vol", "high_vol"}.issubset(vol_values)

    liq_values = set(regimes.loc[regimes["name"] == "liquidity_regime", "value"].unique())
    assert {"dry", "balanced", "flood"}.issubset(liq_values)

    rotation_values = set(regimes.loc[regimes["name"] == "rotation_role", "value"].unique())
    assert {"sector_laggard", "sector_neutral", "sector_leader"}.issubset(rotation_values)

    validated = validate_regime_frame(regimes)
    pdt.assert_frame_equal(regimes.reset_index(drop=True), validated.reset_index(drop=True))


def test_generate_regime_labels_without_turnover_column() -> None:
    price_df = _synthetic_price_panel().drop(columns=["turnover"])
    regimes = generate_regime_labels(price_df)

    liq_values = set(regimes.loc[regimes["name"] == "liquidity_regime", "value"].unique())
    assert {"dry", "balanced", "flood"}.intersection(liq_values)


def test_generate_regime_labels_is_deterministic() -> None:
    price_df = _synthetic_price_panel()
    first = generate_regime_labels(price_df)
    second = generate_regime_labels(price_df)

    pdt.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
