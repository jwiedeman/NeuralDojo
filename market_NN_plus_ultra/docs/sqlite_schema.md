# Market NN Plus Ultra SQLite Schema

This document defines the canonical SQLite contract consumed by the Plus Ultra
agent. The schema is intentionally opinionated to keep ingestion, feature
engineering, and evaluation deterministic while remaining extensible for future
alpha sources.

## Core Tables

### `assets`

| Column        | Type    | Notes                                                   |
|---------------|---------|---------------------------------------------------------|
| `asset_id`    | INTEGER | Primary key. Stable identifier used across tables.      |
| `symbol`      | TEXT    | Human readable ticker (e.g., `AAPL`, `BTC-USD`).        |
| `sector`      | TEXT    | Optional sector/industry label.                         |
| `currency`    | TEXT    | ISO currency code (`USD`, `EUR`, etc.).                 |
| `exchange`    | TEXT    | Optional exchange identifier.                           |
| `metadata`    | JSON    | Free-form JSON for custom attributes.                   |

*Primary key:* (`asset_id`)

### `series`

| Column      | Type    | Notes                                                                            |
|-------------|---------|----------------------------------------------------------------------------------|
| `timestamp` | DATETIME| UTC timestamp for the candle close.                                              |
| `symbol`    | TEXT    | Ticker symbol referencing `assets.symbol`.                                       |
| `open`      | REAL    | Open price.                                                                     |
| `high`      | REAL    | High price.                                                                     |
| `low`       | REAL    | Low price.                                                                      |
| `close`     | REAL    | Close price.                                                                    |
| `volume`    | REAL    | Trade volume.                                                                   |
| `vwap`      | REAL    | Optional volume-weighted average price.                                         |
| `turnover`  | REAL    | Optional traded notional.                                                        |

*Primary key:* (`timestamp`, `symbol`)

*Suggested indices:* (`symbol`, `timestamp` DESC) for fast symbol slicing.

### `indicators`

| Column      | Type    | Notes                                                                      |
|-------------|---------|----------------------------------------------------------------------------|
| `timestamp` | DATETIME| UTC timestamp (must align with `series.timestamp`).                         |
| `symbol`    | TEXT    | Ticker symbol referencing `assets.symbol`.                                 |
| `name`      | TEXT    | Indicator identifier (e.g., `rsi_14`, `onchain_active_addresses`).         |
| `value`     | REAL    | Indicator value.                                                            |
| `metadata`  | JSON    | Optional JSON payload (horizon, parameters, provenance).                    |

*Primary key:* (`timestamp`, `symbol`, `name`)

*Suggested indices:* (`name`, `timestamp`), (`symbol`, `name`, `timestamp`).

### `regimes`

| Column      | Type     | Notes                                                                      |
|-------------|----------|----------------------------------------------------------------------------|
| `timestamp` | DATETIME | UTC timestamp aligned with `series.timestamp`.                             |
| `symbol`    | TEXT     | Ticker symbol referencing `assets.symbol`.                                 |
| `name`      | TEXT     | Regime feature name (`regime_label`, `vol_bucket`, etc.).                   |
| `value`     | TEXT     | Discrete label or JSON blob describing the detected market regime.         |

*Primary key:* (`timestamp`, `symbol`, `name`)

*Suggested indices:* (`symbol`, `timestamp`), (`name`, `timestamp`).

### `trades`

| Column        | Type     | Notes                                                                   |
|---------------|----------|-------------------------------------------------------------------------|
| `trade_id`    | INTEGER  | Primary key.                                                            |
| `timestamp`   | DATETIME | Execution timestamp.                                                    |
| `symbol`      | TEXT     | Ticker symbol referencing `assets.symbol`.                              |
| `side`        | TEXT     | `long` / `short` / `flat`.                                              |
| `size`        | REAL     | Position size in units.                                                 |
| `price`       | REAL     | Fill price.                                                             |
| `fees`        | REAL     | Transaction costs paid (optional).                                      |
| `slippage_bp` | REAL     | Slippage in basis points relative to mid price (optional).              |
| `pnl`         | REAL     | Realised profit and loss associated with the trade (optional).          |
| `metadata`    | JSON     | Optional JSON for strategy identifiers, notes, or scenario tags.        |

*Primary key:* (`trade_id`)

*Suggested indices:* (`symbol`, `timestamp`), (`timestamp` DESC).

### `trade_annotations`

| Column                  | Type     | Notes                                                                                     |
|-------------------------|----------|-------------------------------------------------------------------------------------------|
| `annotation_id`         | INTEGER  | Primary key for the annotation entry.                                                     |
| `trade_id`              | INTEGER  | Foreign key referencing `trades.trade_id`.                                                |
| `symbol`                | TEXT     | Optional symbol override when annotating external trades.                                 |
| `trade_timestamp`       | DATETIME | Timestamp of the underlying trade (mirrors `trades.timestamp`).                           |
| `decision`              | TEXT     | Analyst verdict (`approve`, `reject`, `flag`, `hold`, `escalate`).                         |
| `rationale`             | TEXT     | Free-form explanation describing the decision.                                            |
| `confidence`            | REAL     | Optional analyst confidence score in the range [0, 1].                                    |
| `tags`                  | TEXT     | Comma-separated tags used for downstream filtering.                                      |
| `author`                | TEXT     | Analyst or reviewer responsible for the annotation.                                       |
| `created_at`            | DATETIME | Timestamp (UTC) when the annotation was recorded.                                         |
| `metadata`              | JSON     | Optional structured payload (ticket IDs, escalation references, replay-buffer hints).     |
| `context_window_start`  | DATETIME | Optional UTC timestamp marking the beginning of the reviewed market window.               |
| `context_window_end`    | DATETIME | Optional UTC timestamp marking the end of the reviewed market window.                     |

*Primary key:* (`annotation_id`)

*Suggested indices:* (`trade_id`, `created_at` DESC), (`decision`, `created_at` DESC).

### `benchmarks`

| Column      | Type     | Notes                                                                   |
|-------------|----------|-------------------------------------------------------------------------|
| `timestamp` | DATETIME | UTC timestamp aligned with `series`.                                    |
| `symbol`    | TEXT     | Benchmark identifier (e.g., `SPY`, `BTC-INDEX`).                         |
| `return`    | REAL     | Period-over-period return for the benchmark.                             |
| `level`     | REAL     | Optional benchmark level (e.g., index price).                            |

*Primary key:* (`timestamp`, `symbol`)

### `cross_asset_views`

| Column      | Type     | Notes                                                                                     |
|-------------|----------|-------------------------------------------------------------------------------------------|
| `timestamp` | DATETIME | UTC timestamp aligned with the master series timeline.                                    |
| `feature`   | TEXT     | Feature identifier composed of `<field>__<symbol>` (e.g., `close__SPY`, `log_return_1__QQQ`). |
| `value`     | REAL     | Aligned feature value for the corresponding timestamp/feature pair.                       |
| `universe`  | TEXT     | Optional description of the asset universe that produced the row (comma-separated list).  |
| `metadata`  | JSON     | JSON payload describing the feature (field, symbol) for downstream tooling.               |

*Primary key:* (`timestamp`, `feature`)

## Integrity & Validation

Market NN Plus Ultra ships a dedicated validation layer in
`market_nn_plus_ultra/data/validation.py` that mirrors the schema above with
Pandera models. Each table documented here has a corresponding helper:

| Table | Validation helper |
| --- | --- |
| `assets` | [`validate_assets_frame`](../market_nn_plus_ultra/data/validation.py) |
| `series` | [`validate_price_frame`](../market_nn_plus_ultra/data/validation.py) |
| `indicators` | [`validate_indicator_frame`](../market_nn_plus_ultra/data/validation.py) |
| `regimes` | [`validate_regime_frame`](../market_nn_plus_ultra/data/validation.py) |
| `trades` | [`validate_trades_frame`](../market_nn_plus_ultra/data/validation.py) |
| `benchmarks` | [`validate_benchmark_frame`](../market_nn_plus_ultra/data/validation.py) |
| `cross_asset_views` | [`validate_cross_asset_view_frame`](../market_nn_plus_ultra/data/validation.py) |

These functions wrap explicit Pandera schemas (`ASSET_SCHEMA`, `PRICE_SCHEMA`,
etc.) to enforce types, column presence, and uniqueness constraints while
providing structured error payloads for CLI and pipeline consumers.

Additional guardrails applied during validation:

* Enforce foreign key relationships: `series.symbol`, `indicators.symbol`, and
  `trades.symbol` should reference `assets.symbol` to avoid orphaned records.
* Timestamps must be stored in UTC. Apply timezone conversions during
  ingestion via `SQLiteMarketDataset` if localised views are required.
* Missing values should be represented as `NULL`. The loader forward-fills and
  back-fills per-symbol panels before windowing to avoid artificial gaps.
* Consider storing derived indicators that are expensive to recompute in the
  `indicators` table, while fast-on-the-fly metrics can live inside the feature
  registry.

## Example Queries

Fetch the latest 1,000 candles and indicators for a symbol:

```sql
WITH latest AS (
  SELECT timestamp
  FROM series
  WHERE symbol = 'AAPL'
  ORDER BY timestamp DESC
  LIMIT 1000
)
SELECT s.timestamp, s.open, s.high, s.low, s.close, s.volume,
       i.name AS indicator_name, i.value AS indicator_value
FROM series AS s
LEFT JOIN indicators AS i
  ON s.timestamp = i.timestamp AND s.symbol = i.symbol
WHERE s.symbol = 'AAPL' AND s.timestamp IN latest;
```

Compute realised returns for evaluation joins:

```sql
SELECT t.timestamp,
       t.symbol,
       t.pnl,
       b.return AS benchmark_return
FROM trades AS t
LEFT JOIN benchmarks AS b
  ON t.timestamp = b.timestamp AND b.symbol = 'SPY';
```

Maintaining this contract ensures the Market NN Plus Ultra ingestion pipeline
can spin up feature engineering, training, and inference jobs without manual
data wrangling.

## Fixture Generation & Data Fusion Guidance

The `scripts/make_fixture.py` utility produces long-horizon fixtures that
combine price history, engineered technical indicators, alternative data, and
regime labels into a single SQLite database. The generator follows these steps:

1. **Price synthesis** – create multi-symbol OHLCV series spanning tens of
   thousands of rows per asset using log-normal random walks. Generated frames
   are validated via `validate_price_frame` to guarantee schema compliance.
2. **Technical overlays** – derive moving averages, realised volatility,
   momentum, and drawdown indicators before melting them into the canonical
   `indicators` table. Pandera validation (`validate_indicator_frame`) enforces
   proper typing and duplicate guards.
3. **Alternative data fusion** – add configurable synthetic signals (e.g.
   rolling funding-rate proxies, sentiment momentum) per symbol. Use
   `--alt-features` to control the breadth of signals fused alongside the price
   series.
4. **Regime annotation** – run the quantile-driven labelling pipeline in
   `market_nn_plus_ultra.data.labelling.generate_regime_labels` to compute
   volatility buckets, liquidity states, and cross-sectional rotation roles.
   The resulting multi-dimensional labels are written into the `regimes`
   table and validated alongside the other fixtures so experiments inherit
   deterministic market-state context.
5. **Asset metadata** – seed the `assets` table with synthetic entries so
   foreign keys remain consistent across joins.

Alternative data tables (macro calendars, funding rates, sentiment snapshots,
corporate actions) can now be declared in experiment YAML under
`data.alternative_data`. Each entry maps to an
`AlternativeDataSpec`/`AlternativeDataConnector` pair that automatically joins
the SQLite table on the requested columns, applies optional filters, and
forward/back-fills values so every training window sees a consistent view of
external signals. See `market_nn_plus_ultra.tests.test_alternative_data` for a
minimal example that fuses macro and funding series into the model panel.

Run the generator from the project root to create a GPU-saturating fixture with
30k+ candles per symbol:

```bash
python scripts/make_fixture.py data/plus_ultra_fixture.db \
    --symbols SPY QQQ IWM BTC-USD ETH-USD \
    --rows 32768 --freq 15min --alt-features 4
```

The resulting database slots directly into the training configs via
`data.sqlite_path`. Because every table is validated before persistence, the
fixtures double as documentation for how to fuse long price histories,
technicals, alternative data, and regime context into a reproducible SQLite
asset store.

## Regime Labelling CLI & Troubleshooting

Use the dataset-build CLI to refresh regime annotations and exercise the full
Pandera validation bundle from the terminal:

```bash
python -m market_nn_plus_ultra.cli.dataset_build data/market.db \
    --regime-labels --strict-validation \
    --regime-bands volatility:0.25,0.75 \
    --regime-bands liquidity:0.2,0.8
```

Generate aligned multi-symbol tensors alongside validation by adding the cross-asset flag:

```bash
python -m market_nn_plus_ultra.cli.dataset_build data/market.db \
    --cross-asset-view --cross-asset-columns close volume \
    --cross-asset-fill-limit 32
```

Key troubleshooting tips:

* **Mismatched label counts** – When the CLI detects duplicate or missing
  (`timestamp`, `symbol`, `name`) combinations it logs `label_integrity_error`
  events before aborting. Regenerate labels with `--regime-labels` to rebuild
  the table deterministically, and ensure the requested `--symbol-universe`
  covers every asset referenced in `regimes`.
* **Stale quantile caches** – Adjust the quantile overrides with repeated
  `--regime-bands` flags (e.g. `rotation:0.2,0.8`) to force fresh bands when the
  underlying series distribution shifts. The CLI writes the regenerated table
  atomically so downstream jobs never observe partially updated labels.
* **Strict mode halts** – With `--strict-validation` enabled, Pandera schema and
  foreign-key checks execute after the regime table is replaced. Validation
  failures include structured samples in the log output so you can reconcile the
  offending rows directly in SQLite or regenerate fixtures via
  `scripts/make_fixture.py` before rerunning the CLI.
* **Cross-asset fill limits** – When `--cross-asset-fill-limit` is low and
  source tables contain gaps, the CLI logs `cross_asset_view_written` events that
  include `dropped_rows`/`dropped_features`. Use higher limits or refresh
  fixtures to recover the missing points when the view must remain dense.

