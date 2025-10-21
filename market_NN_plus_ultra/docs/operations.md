# Operations & Monitoring Playbook

This guide documents how to monitor the Market NN Plus Ultra inference service in
production environments. It complements the reporting and automation docs by
focusing on live observability, SLOs, and alert policies.

## Prometheus Metrics

The FastAPI service exposes Prometheus-formatted metrics at `GET /metrics`. The
endpoint is enabled by default and can be scraped by Prometheus or any
compatible collector.

| Metric | Type | Description |
| --- | --- | --- |
| `plus_ultra_service_requests_total{endpoint,method,status}` | Counter | Request volume split by endpoint, method, and HTTP status. |
| `plus_ultra_service_request_latency_seconds{endpoint,method}` | Histogram | Latency distribution for each endpoint. |
| `plus_ultra_service_request_exceptions_total{endpoint,method,exception}` | Counter | Exception counts keyed by exception class. |
| `plus_ultra_service_last_success_timestamp{endpoint}` | Gauge | Unix timestamp of the most recent successful request per endpoint. |
| `plus_ultra_service_last_error_timestamp{endpoint}` | Gauge | Unix timestamp of the most recent failed request per endpoint. |
| `plus_ultra_service_prediction_rows` | Histogram | Distribution of prediction row counts returned by `/predict`. |
| `plus_ultra_service_last_prediction_rows` | Gauge | Row count of the most recent `/predict` response. |
| `plus_ultra_guardrails_enabled{policy}` | Gauge | Indicates whether guardrails are enabled (1) or disabled (0). |
| `plus_ultra_guardrail_violations_total{name}` | Counter | Cumulative guardrail violations by name. |
| `plus_ultra_guardrail_last_violation_count` | Gauge | Number of violations produced by the most recent `/guardrails` call. |

Scrape every 15–30 seconds for real-time visibility. When deploying with
Gunicorn/uvicorn workers ensure the Prometheus client runs in single-process
mode or configure the standard multi-process collector.

## Service Level Objectives

* **Availability:** 99.5% of `/predict` requests must succeed (HTTP status
  `< 500`) over a 7-day rolling window.
* **Latency:** The 95th percentile latency for `/predict` should remain below
  750 ms, and `/guardrails` below 250 ms.
* **Correctness:** Guardrail violations should average fewer than 3 per 1,000
  trades under normal operations. Spikes indicate upstream data or policy
  regressions.

## Alert Policies

| Trigger | Threshold | Action |
| --- | --- | --- |
| Error rate | `rate(plus_ultra_service_requests_total{endpoint="predict",status=~"5.."}[5m]) > 0` | Page the on-call engineer; check model checkpoint health and upstream data freshness. |
| Latency | `histogram_quantile(0.95, sum(rate(plus_ultra_service_request_latency_seconds_bucket{endpoint="predict"}[5m])) by (le)) > 0.75` | Create incident ticket; scale workers or investigate recent deployments. |
| Guardrail violations | `increase(plus_ultra_guardrail_violations_total[30m]) > 10` | Notify trading supervision to review recent trades; consider halting automation. |
| Guardrails disabled | `plus_ultra_guardrails_enabled == 0` for 5 minutes | Alert operations to verify configuration drift or intentional maintenance. |

Integrate alerts with the existing orchestration stack (Dagster/Airflow) so
automation runs can branch or halt when SLOs breach.

## Runbook Notes

1. **Initial triage:** Inspect `/metrics` and the latest `/predict` telemetry
   payloads. Confirm the guardrail gauge still reads `1` and review recent
   `plus_ultra_service_request_exceptions_total` labels.
2. **Data freshness:** Use `scripts/automation/retrain.py --dry-run` to ensure
   retraining triggers are healthy if guardrail violations spike.
3. **Scaling:** Increase uvicorn workers or move to GPU-backed instances if
   latency alerts coincide with throughput growth.
4. **Experiment branch:** When live metrics degrade but data is healthy, branch
   a new experiment run using the most recent stable checkpoint. The
   `plus_ultra_service_last_success_timestamp` gauge should jump once the new
   model is rolled out.
5. **Post-run profitability snapshot:** Kick off
   `scripts/automation/retrain.py --run-evaluation` after long training jobs to
   emit predictions and an operations summary JSON. The evaluation stage writes
   ROI, Sharpe, drawdown, and guardrail alerts to the orchestration run folder
   so operations leads can review profitability alongside guardrail coverage
   before deploying a new checkpoint.

## Human-in-the-loop Overrides & Approvals

The monitoring stack pairs with a lightweight approval workflow so risk teams
can pause, override, or green-light retraining runs before the latest
checkpoint ships to production.

### Pre-approval Checklist

1. **Compile profitability + guardrail metrics** using the automation CLI:
   ```bash
   python scripts/automation/retrain.py \
     --dataset data/market.db \
     --train-config configs/default.yaml \
     --run-evaluation \
     --eval-output automation_runs/$(date +%Y%m%d%H%M)_eval
   ```
   This produces predictions, profitability metrics, and an
   `operations_summary.json` file built from
   `compile_operations_summary()`. Risk teams review the Sharpe, drawdown, and
   guardrail alerts before approving a rollout.
2. **Generate a narrative report** for the review meeting:
   ```bash
   python scripts/generate_report.py \
     --predictions automation_runs/.../predictions.parquet \
     --metrics automation_runs/.../metrics.json \
     --output reports/candidate.html
   ```
   The report packages ROI, attribution, and bootstrapped confidence intervals
   so supervisors can drill into the proposed strategy.
3. **Capture analyst feedback and override context** with
   `scripts/annotate_trades.py`. Every approval or rejection is logged back into
   SQLite and automatically shows up in PPO replay buffers and reporting docs.
4. **Record the decision** in the retraining run folder. Append a short note to
   `operations_summary.json` (or a sibling Markdown file) capturing the reviewer,
   rationale, and escalation path so the automation log stays self-contained.

### Guardrail-driven Decision Tree

* **No alerts triggered:** Promote the checkpoint, schedule the deployment via
  the orchestration layer, and document the rollout in the implementation log.
* **Guardrail alerts but data quality intact:** Engage the risk lead. They may
  green-light with reduced notional caps, request additional PPO warm-starts, or
  block deployment pending further analysis.
* **Data/telemetry anomalies:** Halt deployment, run
  `scripts/monitoring/live_monitor.py` against fresh returns, and initiate the
  data-quality incident playbook before re-running training.

### Escalation Matrix

| Trigger | Escalation Owner | Expected Action |
| --- | --- | --- |
| `operations_summary.json` contains drawdown or Sharpe alerts | Risk lead | Deep dive into guardrail metrics, request simulator what-if analysis, approve/deny rollout |
| Monitoring drift metrics above thresholds | Data engineering | Validate data feeds, re-run dataset build with `--strict-validation`, coordinate with risk lead |
| Service guardrail endpoint returns violations > 0.5% of trades | Trading supervision | Review annotated trade log, enforce manual overrides, notify compliance |
| Analyst rejects trade annotations | Portfolio manager | Evaluate rationale, update strategy notes, feed annotations back into replay buffers |

## Integrated Operations Workflow

The MVP operations stack now merges monitoring, reporting, and guardrail
enforcement into a single review loop:

1. **Retraining scheduler** detects a dataset refresh and executes
   `run_retraining_plan`, producing checkpoints, profitability summaries, and
   evaluation artefacts.
2. **Evaluation stage** writes predictions, ROI metrics, and guardrail alerts.
   When a benchmark is supplied, excess-return diagnostics are included for
   context.
3. **Monitoring snapshot** consumes the latest realised returns via
   `LiveMonitor` and pushes PSI/JS/KS drift metrics plus Sharpe/drawdown gauges
   into Prometheus. Grafana dashboards display the live KPIs alongside the
   evaluation artefacts.
4. **Operations review** combines the generated HTML/Markdown report, the
   `operations_summary.json`, and the monitoring dashboard. Reviewers annotate
   trades or add approval notes, then update `task_tracker.md` with the decision
   link for traceability.
5. **Deployment** proceeds only after the risk lead signs off. The FastAPI
   service reloads the approved checkpoint, the guardrail endpoint remains
   enabled, and Prometheus/alerting continue to track post-deployment health.

Document follow-up observations in the implementation log so future iterations
inherit context on incident response, override decisions, and monitoring
coverage.
