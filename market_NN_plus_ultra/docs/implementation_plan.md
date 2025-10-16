# Implementation Plan

This living plan translates the Market NN Plus Ultra roadmap into concrete engineering milestones. It is intended to complement `task_tracker.md` by providing narrative context, sequencing, and success criteria for the next stages of development.

## Phase 1 — Data & Feature Depth (Weeks 1-2)

**Objectives**

* Expand the feature registry with non-price signals (macro, on-chain, sentiment embeddings).
* Harden data quality gates around the SQLite ingestion path.
* Provide reproducible datasets for benchmarking.

**Milestones**

1. **Alternative data connectors** — Implement pluggable loaders in `market_nn_plus_ultra.data.feature_pipeline` that can pull macro calendars, funding rates, or sentiment series directly from SQLite side tables. Surface the dependencies through the `FeatureSpec` metadata so experiments can toggle them via config.
2. **Schema enforcement** — Extend `market_nn_plus_ultra.data.validation` with pandera models for each joined table and wire the checks into the CLI tooling. Failure criteria should immediately halt training and emit structured logs.
3. **Fixture generation** — Ship scripts under `scripts/` that can synthesise realistic OHLCV panels into SQLite for smoke testing (`scripts/make_fixture.py`) and document the workflow in `docs/sqlite_schema.md`.

**Exit Criteria**

* Feature registry documentation includes at least five new alternative-data signals.
* Loading malformed SQLite data fails fast with actionable errors.
* CI smoke tests run against an auto-generated SQLite fixture.

## Phase 2 — Model Scaling (Weeks 3-4)

**Objectives**

* Benchmark the omni-scale, Mixture-of-Experts, and state-space backbones across asset classes.
* Improve training stability at long context lengths.
* Add calibration-aware output heads.

**Milestones**

1. **Benchmark harness** — Create benchmarking scripts under `scripts/benchmarks/` that sweep architectures, depths, and horizons, storing metrics in a parquet catalogue. Integrate with `docs/research_agenda.md` so results feed back into planning.
2. **Stability tooling** — Implement gradient noise scale diagnostics and loss landscape plots within `market_nn_plus_ultra.training.train_loop`. Persist diagnostics to disk for every experiment.
3. **Calibration head** — Introduce a Dirichlet/quantile hybrid head in `market_nn_plus_ultra.models` with supporting loss terms, enabling probability-calibrated signals for downstream risk controls.

**Exit Criteria**

* Benchmark catalogue contains comparative ROI, Sharpe, and drawdown metrics for at least three model families.
* Training logs include automated alerts when instability is detected (e.g., exploding gradients, poor calibration).
* Inference agent can optionally output calibrated probability distributions.

## Phase 3 — Reinforcement & Automation (Weeks 5-6)

**Objectives**

* Close the loop from supervised predictions to ROI optimisation.
* Prepare infrastructure for continuous retraining and deployment.

**Milestones**

1. **PPO upgrades** — Extend `scripts/rl_finetune.py` with distributed rollouts, curriculum schedules, and reward shaping toggles. Add replay buffers for optional off-policy learning.
2. **Continuous retraining** — Build a Dagster/Airflow-compatible orchestration script under `scripts/automation/` that ingests fresh SQLite dumps, runs training, backtests, and pushes checkpoints to object storage.
3. **Service interface** — Scaffold a FastAPI or gRPC inference service in `market_nn_plus_ultra.service` that mirrors the existing market agent contract but streams the richer Plus Ultra telemetry.

**Exit Criteria**

* Reinforcement fine-tuning experiments show measurable ROI lift over supervised baselines in the benchmark catalogue.
* Automated retraining can be triggered locally with a single CLI command and produces artefacts suitable for deployment.
* Service scaffold responds to inference requests with millisecond latency on CPU and GPU.

## Phase 4 — Operational Excellence (Weeks 7-8)

**Objectives**

* Productionise monitoring, reporting, and guardrails.
* Provide stakeholder-ready reporting and experimentation tools.

**Milestones**

1. **Run reporting** — Extend `scripts/generate_report.py` to incorporate backtest charts, attribution tables, and scenario analysis. Emit both HTML and lightweight Markdown outputs.
2. **Live monitoring** — Integrate Prometheus metrics and alerting hooks into the inference service. Document SLOs and alert policies in `docs/operations.md`.
3. **Risk guardrails** — Build guardrail modules that enforce exposure, turnover, and tail-risk thresholds during inference, configurable via YAML.

**Exit Criteria**

* Reports meet investor presentation quality with minimal manual editing.
* Monitoring dashboard tracks latency, throughput, and ROI drift in real time.
* Guardrail violations trigger structured alerts and optional automatic de-risking actions.

---

### Dependencies & Tooling

* Python ≥3.10, PyTorch 2.2+, PyTorch Lightning 2.2+.
* Optional integrations: Weights & Biases, Pandera, TA-Lib via the `ta` package.
* CI to include `python scripts/ci_compile.py --quiet`, targeted `pytest` suites, and linting once the codebase stabilises.

### Contribution Expectations

* Update `task_tracker.md` with links to experiments and PRs when completing milestones.
* Keep documentation (`docs/`) in sync with new capabilities to reduce onboarding friction.
* Follow the structured logging patterns established in `market_nn_plus_ultra.utils.logging`.

