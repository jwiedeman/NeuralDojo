# Market NN Plus Ultra — End-to-End MVP Quickstart

This guide walks through the minimum viable pipeline needed to let the Plus
Ultra agent learn from a SQLite fixture, evaluate profitability, and export a
monitoring snapshot you can stream into Prometheus. It stitches together the
existing tooling so you can observe ROI, drawdowns, and guardrail alerts after
every run.

## 1. Prepare a reproducible fixture

Start by generating a self-contained SQLite database with prices, indicators,
and regime labels. The fixture ships with enough variance to stress local
training runs while staying small enough for a CPU-only smoke test.

```bash
cd market_NN_plus_ultra
python scripts/make_fixture.py data/plus_ultra_fixture.db \
  --symbols SPY QQQ IWM BTC-USD ETH-USD \
  --rows 32768 --freq 15min --alt-features 4
```

The script validates the output with Pandera before writing to disk. Point all
subsequent commands at `data/plus_ultra_fixture.db` (or adjust the paths if you
already have a richer SQLite dump).

## 2. Launch the retraining plan with evaluation enabled

The automation CLI wires validation, pretraining, supervised tuning, and
(optional) PPO warm starts together. For the MVP we run pretraining, supervised
training, and the evaluation stage with guardrail thresholds so profitability
signals propagate automatically.

```bash
python scripts/automation/retrain.py \
  --dataset data/plus_ultra_fixture.db \
  --train-config configs/default.yaml \
  --pretrain-config configs/pretrain.yaml \
  --run-evaluation \
  --eval-output automation_runs/latest/evaluation \
  --eval-min-sharpe 0.5 \
  --eval-max-drawdown 0.1 \
  --eval-max-gross-exposure 1.0 \
  --eval-max-turnover 1.5
```

> **Tip:** For CPU-only smoke tests or notebook walkthroughs use the lightweight
> configs added under `configs/mvp_pretrain.yaml` and `configs/mvp_quickstart.yaml`.
> They trim model depth, epochs, and dataloader workers so the orchestration run
> completes in minutes while still exercising evaluation + guardrail wiring.

The command prints a summary of each stage. Evaluation artefacts land under the
chosen output directory (default: `automation_runs/<timestamp>/evaluation/`).
Expect to see at least:

* `predictions.parquet` — realised returns with model predictions.
* `operations_summary.json` — ROI, Sharpe, drawdown, guardrails, and triggered
  alerts based on the thresholds above.
* `metrics.json` (when validation metrics are available).

## 3. Capture a reference return series

The monitor compares live performance against a historical baseline. For a
quick smoke test, export the validation-split realised returns from the
predictions file itself:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

predictions = pd.read_parquet(
    Path("automation_runs/latest/evaluation/predictions.parquet")
)
reference = predictions[["realised_return"]].dropna()
reference.to_parquet("automation_runs/latest/evaluation/reference_returns.parquet")
PY
```

For production you would swap this step with a curated backtest or benchmark
returns catalogue.

## 4. Produce a monitoring snapshot

With the reference series and evaluation artefacts in place, call the monitoring
CLI. It now understands the evaluation directory and merges profitability data
with drift diagnostics automatically.

```bash
python scripts/monitoring/live_monitor.py \
  automation_runs/latest/evaluation/reference_returns.parquet \
  --evaluation-dir automation_runs/latest/evaluation \
  --output automation_runs/latest/evaluation/monitoring_snapshot.json
```

The command prints risk, drift, and guardrail tables, lists any alerts, and
writes a JSON payload. You can feed the same snapshot into the Prometheus
exporter:

```bash
python - <<'PY'
from prometheus_client import CollectorRegistry, push_to_gateway
from market_nn_plus_ultra.monitoring.prometheus import make_prometheus_exporter
from market_nn_plus_ultra.monitoring.live import MonitoringSnapshot, DriftMetrics
import json
from pathlib import Path

payload = json.loads(Path("automation_runs/latest/evaluation/monitoring_snapshot.json").read_text())
registry = CollectorRegistry()
exporter = make_prometheus_exporter(registry)

snapshot = MonitoringSnapshot(
    risk=payload["risk"],
    drift=DriftMetrics(**payload["drift"]),
    alerts=payload["alerts"],
    window_count=payload["window_count"],
    guardrails=payload.get("guardrails"),
)
exporter.publish(snapshot)
push_to_gateway("http://localhost:9091", job="plus-ultra-mvp", registry=registry)
PY
```

Replace the Pushgateway endpoint with your deployment target or reuse the
registry directly inside a long-running service.

## 5. Iterate

* Tune the thresholds passed to `retrain.py` to align with your risk appetite.
* Enable PPO with `--run-reinforcement` and `--warm-start training` once the MVP
  loop is stable.
* Drop the reference-generation helper once you have a curated baseline stored
  alongside evaluation artefacts.

Following these steps gives you a single command (`retrain.py`) that trains the
agent, evaluates profitability, and leaves behind everything required to monitor
Sharpe, drawdowns, and guardrail breaches after each run.

For a guided, executable version of this loop, open
`notebooks/mvp_retraining_walkthrough.ipynb`. The notebook keeps the quickstart
checklist intact, runs the orchestration pipeline end-to-end, and saves the
resulting artefacts under `automation_runs/mvp_notebook/` for review.
