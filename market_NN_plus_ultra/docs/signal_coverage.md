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
