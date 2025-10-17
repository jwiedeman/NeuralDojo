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

### `benchmarks`

| Column      | Type     | Notes                                                                   |
|-------------|----------|-------------------------------------------------------------------------|
| `timestamp` | DATETIME | UTC timestamp aligned with `series`.                                    |
| `symbol`    | TEXT     | Benchmark identifier (e.g., `SPY`, `BTC-INDEX`).                         |
| `return`    | REAL     | Period-over-period return for the benchmark.                             |
| `level`     | REAL     | Optional benchmark level (e.g., index price).                            |

*Primary key:* (`timestamp`, `symbol`)

## Integrity & Validation

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
4. **Regime annotation** – bucket forward returns into bear/neutral/bull
   classes and persist them inside the dedicated `regimes` table. Downstream
   experiments can join these labels to condition models on market state.
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

