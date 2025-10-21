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

Document follow-up observations in the implementation log so future iterations
inherit context on incident response and monitoring coverage.
