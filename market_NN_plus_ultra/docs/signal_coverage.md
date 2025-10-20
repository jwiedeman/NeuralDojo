# Signal Coverage, Cross-Asset Views, and Deterministic Tooling

This document inventories the coverage provided by the Market NN Plus Ultra
dataset tooling across raw prices, engineered signals, and fused alternative
data sources.

## Cross-Asset Feature Views

The dataset builder ships an optional cross-asset tensor that aligns
multi-symbol features into a single long-form table (`cross_asset_views`). The
routine lives in [`market_nn_plus_ultra.data.cross_asset.build_cross_asset_view`](../market_nn_plus_ultra/data/cross_asset.py)
and is exposed through the CLI entry point
[`market_nn_plus_ultra.cli.dataset_build`](../market_nn_plus_ultra/cli/dataset_build.py).

Pass the following flags to customise the output when running
`market-nn-plus-ultra-dataset-build`:

* `--cross-asset-view` – materialise the `cross_asset_views` table using the
  current `series` data.
* `--cross-asset-columns` – choose which OHLCV fields to align (defaults to
  `close` and `volume`).
* `--cross-asset-fill-limit` – cap the number of consecutive missing values that
  will be forward/back filled when stitching timelines.
* `--cross-asset-no-returns` – skip automatic log-return features to keep the
  view restricted to the selected raw columns.

When strict validation is enabled (`--strict-validation`) the resulting table is
verified via [`validate_cross_asset_view_frame`](../market_nn_plus_ultra/data/validation.py)
which enforces the Pandera schema and uniqueness guarantees documented in
[`sqlite_schema.md`](sqlite_schema.md).

## Alternative Data Pairings

`SQLiteMarketDataset` pairs auxiliary tables through
[`AlternativeDataSpec`](../market_nn_plus_ultra/data/alternative_data.py) definitions,
allowing a mixture of macro calendars, derivative funding rates, or custom
signals to be merged alongside prices. Regression coverage in
[`tests/test_alternative_data.py`](../tests/test_alternative_data.py) ensures the
join semantics are deterministic: forward-filled values for sparse timestamps
remain stable across runs and each spec prefixes its columns consistently.

### Pairing Matrix for Long-Horizon Experiments

The optimisation plan calls for coupling technical indicators with
alternative-data feeds so multi-quarter experiments ingest complementary
signals. The following matrix captures the supported pairings and the CLI
toggles needed to activate them:

| Alternative Source | Technical Bundle | CLI / Config Toggle | Notes |
| --- | --- | --- | --- |
| Macro calendar surprises (`macro_calendar` table) | Multi-horizon moving averages + volatility bands | `--feature-set macro_ma_vol` | Aligns macro releases with volatility responses; relies on regime labels to gate post-event drift windows. |
| On-chain activity (exchange flows, active addresses) | Momentum + order-book imbalance | `--feature-set onchain_momo_obi` | Designed for crypto universes; combines alternative data lag features with cross-asset ETF hedges when `--cross-asset-view` is enabled. |
| Funding rates & basis spreads | Carry & term-structure factors | `--feature-set funding_carry_term` | Couples futures funding with yield-curve derived PCA factors for duration-aware carry trades. |
| Corporate actions & buyback cadence | Relative strength & liquidity ranks | `--feature-set corp_rslr` | Useful for equities; deterministic joins keep split/dividend adjustments aligned with liquidity filters. |

Each bundle ships with validation hooks under
[`market_nn_plus_ultra.data.validation`](../market_nn_plus_ultra/data/validation.py)
to guarantee the merged frame satisfies the same null, duplicate, and key
integrity rules as the base tables. When running the dataset builder, combine
`--strict-validation` with the appropriate `--feature-set` entry so the
expanded joins fail fast if dependencies drift.

### Regime-Aware Pairings

The new bundles integrate directly with the regime labelling pipeline. By
enabling `--regime-labels` alongside a pairing, the dataset builder will attach
regime IDs and the corresponding embedding inputs so downstream training jobs
can filter or weight samples by market state. This is especially useful for
long-horizon runs where alternative-data signals exhibit regime-dependent
predictive power.

For reproducibility, recommended workflows:

1. Run `market-nn-plus-ultra-dataset-build --feature-set <bundle> --regime-labels --strict-validation` to materialise the fused dataset.
2. Inspect the generated Markdown report from the dataset build (see
   [`docs/reporting.md`](reporting.md)) to confirm signal coverage and guardrail
   metrics.
3. Record the resulting bundle name in experiment YAML under
   `feature_pipeline.active_bundles` so training and evaluation runs remain in
   sync with the dataset artefacts.

## Regime Labelling Determinism

Regime generation uses
[`generate_regime_labels`](../market_nn_plus_ultra/data/labelling.py) with
quantile-driven breakpoints that can be overridden via the CLI by combining
`--regime-labels` with one or more `--regime-bands` entries. The default
breakpoints and override handling are validated through
[`tests/test_labelling.py`](../tests/test_labelling.py), which asserts
reproducible labels, absence of duplicate keys, and compatibility with
[`validate_regime_frame`](../market_nn_plus_ultra/data/validation.py).

Together, these guarantees let teams toggle advanced artefacts (cross-asset
views, alternative data pairings, and regime relabelling) while retaining a
predictable contract for downstream modelling.
