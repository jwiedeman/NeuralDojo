# Performance Reporting

The Market NN Plus Ultra stack now ships with a lightweight reporting utility
that turns model predictions into investor-friendly summaries. Reports combine
profitability snapshots, attribution tables, scenario analysis, risk metrics,
equity curves, return distributions, and bootstrapped confidence intervals to
highlight how a strategy behaves across time.

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
| `--milestones` | JSON file referencing research agenda milestones to surface in the generated narrative. |
| `--return-column` | Which column represents realised returns; defaults to `realised_return`. |
| `--no-equity` / `--no-distribution` | Disable chart generation to speed up quick summaries. |
| `--charts-dir-name` | Custom name for the folder that stores generated charts (defaults to `<output>_assets`). |

Reports automatically create a sibling folder (e.g. `latest_assets/`) that stores
chart images referenced by the Markdown or HTML output. Both formats are
designed to drop directly into research notebooks, investor letters, or PRDs.

## Report Contents

Every generated report contains the following sections:

* **Overview** — dataset size, symbols covered, and backtest window.
* **Risk Metrics** — Sharpe, Sortino, drawdown, volatility, and other ROI
  diagnostics.
* **Profitability Summary** — total and annualised returns, volatility,
  hit rate, best/worst periods, and trade counts.
* **Attribution by Symbol** — per-ticker sample weights, average returns,
  cumulative PnL, and contribution percentages so desk leads can spot the
  assets driving performance or losses.
* **Scenario Analysis** — worst drawdown window plus best and worst periods
  with timestamps and symbols for quick drill-downs.
* **Bootstrapped Confidence Intervals** — 95% intervals for total return,
  annualised return, and Sharpe derived from thousands of bootstrap resamples.
* **Visualisations** — equity and return-distribution charts saved alongside
  the report.

When milestones are supplied—either via the CLI flag or the Python API—the report
adds a **Research Agenda Alignment** section that links the run back to the
living roadmap in `docs/research_agenda.md`. Each milestone entry supports
`phase`, `milestone`, and optional `summary` fields, making it easy to annotate
reports with the roadmap slice they unlock.

## Training Run Summaries

`python scripts/train.py ...` now emits lightweight profitability summaries for
every supervised training run. Once validation completes, the Lightning module
aggregates ROI, Sharpe, maximum drawdown, scenario highlights, and confidence
intervals across the epoch and writes two artefacts next to the configured
checkpoint directory:

* `profitability_summary.json` — machine-friendly metrics for automation.
* `profitability_summary.md` — a Markdown table ready for experiment notebooks
  or status updates.

The same metrics are exposed via `TrainingRunResult.profitability_summary`, so
benchmark sweeps and follow-on tooling can merge the diagnostics into their own
reports without reloading checkpoints.

### Experiment Tracking

Training and pretraining runs can stream metrics directly to Weights & Biases.
Set `wandb_project` (and optionally `wandb_entity`, `wandb_run_name`, or
`wandb_tags`) in your experiment YAML or pass overrides such as
`--wandb-project plus-ultra --wandb-tags curriculum,long-context` to the CLI.
Offline environments are supported via `wandb_offline: true` or the
`--wandb-offline` flag, which stores run artefacts locally until they are synced.
Each run automatically receives the full experiment configuration so dashboards
are reproducible and searchable.

### Guardrail Diagnostics

The evaluation package now exposes `guardrail_metrics`, which surfaces
exposure, turnover, and tail-risk diagnostics from trade logs. These metrics
are ideal for automated run checks before promoting a strategy:

```python
from market_nn_plus_ultra.evaluation import guardrail_metrics

guardrails = guardrail_metrics(trade_log_df, capital_base=5_000_000)
print(guardrails["gross_exposure_peak"], guardrails["tail_return_quantile"])
```

Combine guardrails with the standard ROI metrics to build dashboards that catch
over-levered policies or fat-tailed behaviour before they reach production.

## Operations Readiness Summary

`compile_operations_summary` brings risk metrics and guardrail diagnostics into
one payload that operations teams can consume before approving a rollout. Feed
it the evaluation returns plus an optional trade log and, when desired, a set of
thresholds that codify your guardrails:

```python
from market_nn_plus_ultra.evaluation import (
    OperationsThresholds,
    compile_operations_summary,
)

summary = compile_operations_summary(
    predictions_df,
    trades_df,
    capital_base=5_000_000,
    thresholds=OperationsThresholds(
        min_sharpe=1.0,
        max_drawdown=0.05,
        max_gross_exposure=0.75,
        max_turnover=0.5,
        min_tail_return=-25_000.0,
        max_tail_frequency=0.1,
    ),
)

if summary.triggered:
    print("Run blocked:")
    for alert in summary.triggered:
        print(" -", alert)
```

`OperationsSummary.as_dict()` returns a serialisable payload that downstream
automation (reports, dashboards, approval workflows) can persist alongside other
telemetry.

## Python API

For programmatic access import the helper from the evaluation package:

```python
from market_nn_plus_ultra.evaluation import generate_report

report_path = generate_report(
    predictions_df,
    "reports/alpha_study.md",
    title="Alpha Study",
    description="30-day walk-forward evaluation across the top 20 symbols.",
    milestones=[
        {
            "phase": "Phase 3 — Evaluation & Monitoring",
            "milestone": "Publish automated run reports",
            "summary": "Walk-forward study wired into the reporting automation backlog.",
        }
    ],
)
```

The function inspects the output suffix to choose Markdown (`.md`) or HTML
(`.html`). Pass `metrics={"benchmark_return": 0.12}` to attach bespoke
statistics.

