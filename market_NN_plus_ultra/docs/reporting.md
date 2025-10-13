# Performance Reporting

The Market NN Plus Ultra stack now ships with a lightweight reporting utility
that turns model predictions into investor-friendly summaries. Reports combine
risk metrics, equity curves, and return distributions to highlight how a
strategy behaves across time.

## CLI Usage

Run the helper after producing a predictions file (CSV, Parquet, JSON) from the
agent:

```bash
python scripts/generate_report.py \
  --predictions outputs/predictions.parquet \
  --output reports/latest.html \
  --title "Crypto Swing Trades — April" \
  --description "Generated from walk-forward backtests on BTC/ETH pair."
```

Key flags:

| Flag | Description |
| --- | --- |
| `--predictions` | Path to a predictions table produced by `run_agent.py` or custom backtests. |
| `--output` | Report destination. Use `.md` for Markdown or `.html` for a styled web page. |
| `--metrics` | Optional JSON file containing extra metrics (e.g. benchmark stats) to merge into the report. |
| `--return-column` | Which column represents realised returns; defaults to `realised_return`. |
| `--no-equity` / `--no-distribution` | Disable chart generation to speed up quick summaries. |

Reports automatically create a sibling folder (e.g. `latest_assets/`) that stores
chart images referenced by the Markdown or HTML output. Both formats are
designed to drop directly into research notebooks, investor letters, or PRDs.

## Python API

For programmatic access import the helper from the evaluation package:

```python
from market_nn_plus_ultra.evaluation import generate_report

report_path = generate_report(
    predictions_df,
    "reports/alpha_study.md",
    title="Alpha Study",
    description="30-day walk-forward evaluation across the top 20 symbols.",
)
```

The function inspects the output suffix to choose Markdown (`.md`) or HTML
(`.html`). Pass `metrics={"benchmark_return": 0.12}` to attach bespoke
statistics.

