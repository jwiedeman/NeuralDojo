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
| `--milestones` | JSON file referencing research agenda milestones to surface in the generated narrative. |
| `--return-column` | Which column represents realised returns; defaults to `realised_return`. |
| `--no-equity` / `--no-distribution` | Disable chart generation to speed up quick summaries. |
| `--charts-dir-name` | Custom name for the folder that stores generated charts (defaults to `<output>_assets`). |

Reports automatically create a sibling folder (e.g. `latest_assets/`) that stores
chart images referenced by the Markdown or HTML output. Both formats are
designed to drop directly into research notebooks, investor letters, or PRDs.

When milestones are supplied—either via the CLI flag or the Python API—the report
adds a **Research Agenda Alignment** section that links the run back to the
living roadmap in `docs/research_agenda.md`. Each milestone entry supports
`phase`, `milestone`, and optional `summary` fields, making it easy to annotate
reports with the roadmap slice they unlock.

## Training Run Summaries

`python scripts/train.py ...` now emits lightweight profitability summaries for
every supervised training run. Once validation completes, the Lightning module
aggregates ROI, Sharpe, and maximum drawdown across the epoch and writes two
artefacts next to the configured checkpoint directory:

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

