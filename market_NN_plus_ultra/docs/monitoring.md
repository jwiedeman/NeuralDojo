# Live Monitoring for Market NN Plus Ultra

The live monitoring stack closes the gap between offline evaluation and
production telemetry by exposing real-time risk and drift diagnostics.
These utilities power the "online monitoring" milestone in the optimisation
plan and integrate with Prometheus so operations teams can alert on
production degradation.

## Components

| Component | Location | Purpose |
| --- | --- | --- |
| `LiveMonitor` | `market_nn_plus_ultra.monitoring.live` | Maintains a rolling window of realised returns and produces risk + drift metrics. |
| `DriftMetrics` | `market_nn_plus_ultra.monitoring.drift` | Computes population stability index, Jensen–Shannon divergence, and Kolmogorov–Smirnov statistics. |
| `PrometheusExporter` | `market_nn_plus_ultra.monitoring.prometheus` | Pushes monitoring snapshots into Prometheus gauges and counters. |
| CLI | `scripts/monitoring/live_monitor.py` | Runs the monitor over recent returns and prints/exports the snapshot. |

## Quickstart

Generate a baseline reference returns file (e.g. from backtests) and a live
returns dump. Then run the monitoring CLI:

```bash
python scripts/monitoring/live_monitor.py \
  data/reference_returns.parquet \
  data/live_returns.parquet \
  --return-col realised_return \
  --window 512 \
  --drift-bins 20 \
  --min-sharpe 0.75 \
  --max-drawdown 0.08 \
  --psi-alert 0.25 \
  --js-alert 0.07 \
  --ks-alert 0.2 \
  --output monitoring_snapshot.json
```

The script prints a summary table to stdout and optionally writes the
snapshot JSON. Prometheus exporters can call
`market_nn_plus_ultra.monitoring.default_prometheus_exporter()` and feed it
the resulting snapshot to keep dashboards up to date.

### Using evaluation artefacts from `run_retraining_plan`

When the automation pipeline finishes with the evaluation stage enabled, the
`evaluation/` directory contains both the predictions parquet and the
operations summary JSON. The monitoring CLI now understands these artefacts
directly, so you can produce a Prometheus-ready snapshot in one command:

```bash
python scripts/monitoring/live_monitor.py \
  automation_runs/20251203-120000/evaluation/reference_returns.parquet \
  --evaluation-dir automation_runs/20251203-120000/evaluation \
  --output automation_runs/20251203-120000/evaluation/monitoring_snapshot.json
```

In this example the reference file holds a baseline return history (e.g. a
validation backtest) stored alongside the evaluation artefacts. `--evaluation-dir`
resolves the predictions (`predictions.parquet` by default) and operations
summary (`operations_summary.json`). The CLI merges the profitability metrics
and guardrail alerts captured during evaluation with live drift checks so the
resulting snapshot includes everything the Prometheus exporter needs. Provide
custom filenames with `--evaluation-predictions-name` or
`--evaluation-operations-name` when the defaults are not used.

## Alert Semantics

* **Risk thresholds** reuse the `OperationsThresholds` dataclass so the
  monitoring alerts match production guardrails (Sharpe, drawdown, VaR, etc.).
* **Drift thresholds** default to conservative values suitable for early
  warning dashboards. Override the PSI/JS/KS limits in the CLI or when
  constructing `LiveMonitor` for bespoke regimes.
* **Prometheus gauges** expose the latest metric values, a window size gauge
  for sanity checks, and counters/timestamps for alert auditing. Negative
  risk metrics (e.g. max drawdown, expected shortfall) are exported as
  absolute magnitudes so dashboard panels remain non-negative while the
  underlying snapshot preserves the sign for offline analysis.

## Next Steps

* Wire the CLI into the retraining scheduler to publish snapshots after each
  deployment.
* Extend the exporter with Grafana-ready JSON payloads when the operations
  playbook unifies reporting + monitoring.
