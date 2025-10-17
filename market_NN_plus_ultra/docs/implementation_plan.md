# Implementation Plan

This living plan translates the Market NN Plus Ultra roadmap into concrete engineering milestones. It is intended to complement `task_tracker.md` by providing narrative context, sequencing, and success criteria for the next stages of development.

## Optimization Execution Log

* **2024-02-25 — Plan resync:** Reconfirmed Phase 1 as the primary optimisation bottleneck. The focus is to finish schema enforcement and unlock richer market-regime labels before scaling benchmarking sweeps.
* **2024-02-25 — Traceability update:** Added implementation notes directly to each milestone below so in-flight work, blockers, and measurement hooks are visible without cross-referencing other docs.
* **Next checkpoint:** Capture concrete telemetry requirements for stability tooling once schema validation lands (target: next iteration review).

## Phase 1 — Data & Feature Depth (Weeks 1-2)

**Objectives**

* Expand the feature registry with non-price signals (macro, on-chain, sentiment embeddings) so the models ingest long historical regimes alongside technical indicators.
* Harden data quality gates around the SQLite ingestion path with strict Pandera validation before training begins.
* Provide reproducible, GPU-ready datasets that maximise signal variance for extended 4090 runs.
* Curate survivorship-bias-free ETF and equity panels with corporate actions, liquidity metrics, and point-in-time fundamentals sourced from internal SQLite snapshots (no third-party API keys required).
* Add governance around data augmentation so multi-resolution indicators, regime labels, and cross-asset context features can be generated deterministically.

**Milestones**

1. **Alternative data connectors** — Implement pluggable loaders in `market_nn_plus_ultra.data.feature_pipeline` that can pull macro calendars, funding rates, sentiment series, options-implied volatility, and corporate action adjustments directly from SQLite side tables. Surface the dependencies through the `FeatureSpec` metadata so experiments can toggle them via config. **Status:** ✅ Completed via the `AlternativeDataSpec` / `AlternativeDataConnector` pipeline and the `data.alternative_data` YAML hook (`market_nn_plus_ultra.data.alternative_data`, `tests/test_alternative_data.py`).
   - *Notes (2024-02-25):* Capturing integration tests that exercise the connectors alongside forthcoming schema validators so new tables inherit validation guarantees once Pandera models land.
2. **Schema enforcement** — Extend `market_nn_plus_ultra.data.validation` with pandera models for each joined table and wire the checks into the CLI tooling. Failure criteria should immediately halt training and emit structured logs, ensuring clean, fused datasets before scaling parameters. **Status:** ⏳ In progress — drafting Pandera models for the `assets`, `series`, and `indicators` tables with reusable validators before integrating them into the CLI runners.
   - *Next steps (2024-02-25):* Finalise cross-table foreign-key assertions, codify null/duplicate rules, and add CLI smoke tests that fail fast when fixtures drift. This unlocks safe execution of optimisation sweeps.
3. **Fixture generation** — Ship scripts under `scripts/` that can synthesise realistic OHLCV panels with long histories into SQLite for smoke testing (`scripts/make_fixture.py`) and document the workflow in `docs/sqlite_schema.md`. Generate reproducible fixtures that saturate GPU training with high-variance signals. **Status:** ✅ Completed via `scripts/make_fixture.py` and the expanded fusion guidance in `docs/sqlite_schema.md`. Notes captured in `docs/sqlite_schema.md#fixture-generation--data-fusion-guidance`.
   - *Notes (2024-02-25):* Queueing a refresh of the synthetic fixtures once schema enforcement is merged so benchmarking datasets reflect the stricter validation rules.
4. **Market-regime labelling** — Build deterministic pipelines in `market_nn_plus_ultra.data.labelling` that compute volatility regimes, liquidity regimes, and sector rotation markers using the enriched feature store so downstream models can condition on market state. **Status:** 🗓 Planned — collecting candidate volatility bands and liquidity heuristics before codifying into reusable label transforms.
   - *Preparation (2024-02-25):* Drafting volatility band heuristics aligned with the optimisation focus on calibration heads; labelling design will piggyback on new Pandera contracts for reliability.
5. **Cross-asset feature views** — Extend dataset assembly scripts to output aligned multi-ticker tensors (e.g., sector ETFs, index futures) that let the policy attend to correlations without requiring live data pulls. **Status:** 🗓 Planned — evaluating join strategies for synchronising ETF sector panels with the core ticker timelines while staying SQLite-friendly.
   - *Preparation (2024-02-25):* Documenting join performance benchmarks required for the optimisation sweep harness so we can measure feature-assembly cost versus training throughput.

**Exit Criteria**

* Feature registry documentation includes at least five new alternative-data signals fused with technical indicators and cross-asset context views.
* Loading malformed SQLite data fails fast with actionable errors and clear Pandera traces.
* CI smoke tests run against an auto-generated SQLite fixture sized for long-horizon experiments and containing regime labels.

## Phase 2 — Model Scaling (Weeks 3-4)

**Objectives**

* Benchmark the omni-scale, Mixture-of-Experts, and state-space backbones across asset classes on a 24 GB RTX 4090.
* Improve training stability at long context lengths through automated diagnostics.
* Add calibration-aware output heads that stay GPU friendly at scale.
* Prototype multi-scale temporal encoders that jointly process intraday, daily, and weekly windows.
* Introduce cross-asset attention and volatility embedding modules that condition actions on market state.

**Milestones**

1. **Benchmark harness** — Create benchmarking scripts under `scripts/benchmarks/` that automate sweeps over architecture type, depth, horizon, dilation schedules, and temporal resolutions, storing metrics in a parquet catalogue. Integrate with `docs/research_agenda.md` so results feed back into planning and highlight optimal 4090 utilisation. **Status:** ✅ Completed — `scripts/benchmarks/architecture_sweep.py` now orchestrates configurable sweeps, disables W&B by default for lab runs, and emits parquet catalogues summarising metrics + dataset stats for downstream analysis.
   - *Notes (2024-02-25):* Scheduling reruns post-schema enforcement to validate training throughput deltas once datasets are regenerated.
2. **Stability tooling** — Implement gradient noise scale diagnostics, calibration drift monitoring, and loss landscape plots within `market_nn_plus_ultra.training.train_loop`. Persist diagnostics to disk for every experiment and surface warnings in the CLI. **Status:** 🗓 Planned — reviewing Lightning callback hooks to decide where to insert gradient-noise trackers and calibration alerts.
   - *Next steps (2024-02-25):* Prototype callback scaffolding while awaiting validated datasets so optimisation metrics (noise scale, calibration drift) have trustworthy baselines.
3. **Calibration head** — Introduce a Dirichlet/quantile hybrid head in `market_nn_plus_ultra.models` with supporting loss terms, enabling probability-calibrated signals for downstream risk controls and readying models for reinforcement fine-tuning. **Status:** 🗓 Planned — surveying prior experiments to determine default concentration priors before coding the head.
   - *Preparation (2024-02-25):* Collecting empirical priors from current supervised checkpoints to parameterise the calibration head once optimisation diagnostics are ready.
4. **Multi-scale backbone experiments** — Add hierarchical Transformer and state-space variants in `market_nn_plus_ultra.models.backbones` that fuse intraday/daily/weekly encoders, reporting their contribution to Sharpe and drawdown metrics. **Status:** 🗓 Planned — earmarking architecture configs to prototype once benchmark harness skeleton lands.
   - *Preparation (2024-02-25):* Aligning config templates with optimisation sweep requirements (depth, dilation, device budgeting) so experiments start with consistent measurement hooks.
5. **Market-state embeddings** — Train volatility and regime embedding heads that feed into the policy layers, ensuring the configuration system can toggle them for ablation studies. **Status:** 🗓 Planned — designing embedding interfaces compatible with the forthcoming labelling pipeline.
   - *Notes (2024-02-25):* Waiting on market-regime labelling outputs to set embedding dimensionality and regularisation schedules that will feed optimisation metrics.

**Exit Criteria**

* Benchmark catalogue contains comparative ROI, Sharpe, and drawdown metrics for at least three model families with sweeps across depth, horizon, dilation, and temporal scale settings.
* Training logs include automated alerts when instability is detected (e.g., exploding gradients, poor calibration, gradient noise spikes).
* Inference agent can optionally output calibrated probability distributions that remain compatible with 24 GB VRAM constraints and leverage volatility embeddings.

## Phase 3 — Reinforcement & Automation (Weeks 5-6)

**Objectives**

* Close the loop from supervised predictions to ROI optimisation, using masked/contrastive pretraining as a warm start for large runs.
* Prepare infrastructure for continuous retraining and deployment with curriculum-driven reinforcement learning schedules.
* Simulate realistic execution, slippage, and capital constraints so reinforcement policies optimise for deployable performance.
* Extend objectives with Sharpe, Sortino, drawdown, and CVaR penalties to prioritise risk-adjusted profitability.

**Milestones**

1. **PPO upgrades** — Extend `scripts/rl_finetune.py` with distributed rollouts, curriculum schedules, reward shaping toggles, and hooks that warm-start from the masked/contrastive pretraining checkpoints. Add replay buffers for optional off-policy learning and entropic-risk penalties. **Status:** 🗓 Planned — waiting on calibration diagnostics from Phase 2 to tune curriculum milestones.
   - *Notes (2024-02-25):* Drafting rollout profiling checklist so PPO upgrades inherit the optimisation metrics (latency, gradient noise) prepared in Phase 2.
2. **Execution simulator** — Implement a vectorised trading simulator under `market_nn_plus_ultra.simulation` that models slippage, partial fills, funding costs, latency buckets, and capital constraints using historical limit-order approximations stored locally. **Status:** 🗓 Planned — compiling execution constraints gathered from prior NeuralArena deployments for simulator parameter defaults.
   - *Preparation (2024-02-25):* Mapping simulator telemetry outputs to the reporting stack to keep optimisation and deployment KPIs aligned.
3. **Risk-aware objectives** — Add composite Sharpe/Sortino/drawdown/CVaR losses to the reinforcement training loop with configuration flags to tune their weights, ensuring compatibility with supervised warm starts. **Status:** 🗓 Planned — identifying reusable loss components shared with supervised objectives to avoid duplicate code.
   - *Next steps (2024-02-25):* Reuse optimisation-ready loss components from supervised training to avoid divergence between ROI metrics across regimes.
4. **Continuous retraining** — Build a Dagster/Airflow-compatible orchestration script under `scripts/automation/` that ingests fresh SQLite dumps, runs pretraining warm starts, supervised tuning, reinforcement fine-tuning, and backtests before pushing checkpoints to object storage. **Status:** 🗓 Planned — sketching DAG boundaries and artefact contracts in anticipation of simulator completion.
   - *Notes (2024-02-25):* Identify schedule hooks where optimisation dashboards can plug into orchestration alerts for drift detection.
5. **Service interface** — Scaffold a FastAPI or gRPC inference service in `market_nn_plus_ultra.service` that mirrors the existing market agent contract, streams the richer Plus Ultra telemetry, and exposes curriculum configuration for reinforcement fine-tuning. **Status:** 🗓 Planned — pencilling API parity requirements with the current market agent before implementation.
   - *Preparation (2024-02-25):* Defining optimisation KPIs (latency, throughput, calibration) that must be exposed by the service for Phase 4 monitoring.

**Exit Criteria**

* Reinforcement fine-tuning experiments show measurable ROI lift over supervised baselines in the benchmark catalogue while maintaining drawdown guardrails and respecting simulated execution constraints.
* Automated retraining can be triggered locally with a single CLI command and produces artefacts suitable for deployment, including curriculum and warm-start metadata plus simulator calibration reports.
* Service scaffold responds to inference requests with millisecond latency on CPU and GPU and can trigger PPO fine-tuning loops post-supervised convergence.

## Phase 4 — Operational Excellence (Weeks 7-8)

**Objectives**

* Productionise monitoring, reporting, and guardrails with automated profitability diagnostics.
* Provide stakeholder-ready reporting and experimentation tools for deciding when to extend or branch long-running experiments.
* Establish continuous evaluation, human-in-the-loop review, and alerting around feature drift, regime shifts, and guardrail breaches.

**Milestones**

1. **Run reporting** — Extend `scripts/generate_report.py` to incorporate backtest charts, attribution tables, scenario analysis, profitability summaries, and bootstrapped confidence intervals so every long training session outputs actionable ROI, Sharpe, and drawdown diagnostics. Emit both HTML and lightweight Markdown outputs. **Status:** 🗓 Planned — gathering charting requirements from existing reports to ensure compatibility once monitoring lands.
2. **Live monitoring** — Integrate Prometheus metrics and alerting hooks into the inference service. Document SLOs and alert policies in `docs/operations.md`, including live decision support for extending or branching experiments. **Status:** 🗓 Planned — deferring until the service scaffold from Phase 3 establishes telemetry contracts.
3. **Risk guardrails** — Build guardrail modules that enforce exposure, turnover, tail-risk thresholds, and sector/factor caps during inference, configurable via YAML, and aligned with reinforcement fine-tuning outputs. **Status:** 🗓 Planned — aligning guardrail thresholds with upcoming calibration head outputs.
4. **Analyst feedback loop** — Add review tooling in `market_nn_plus_ultra.reporting` that lets human supervisors annotate trades, flag anomalies, and record veto rationales stored in SQLite for future offline RL. **Status:** 🗓 Planned — capturing analyst workflow requirements while reinforcement hooks are designed.

### Progress Notes — 2024-02-24

* Kicked off Pandera schema enforcement work; drafting validators for core tables and planning CLI integration.
* Documented preparatory research for market-regime labelling and cross-asset feature alignment to unblock subsequent automation tasks.
* Captured pre-work for benchmarking, stability tooling, and downstream reinforcement/operations milestones so sequencing remains intact while optimisations proceed.
* Delivered the first benchmarking harness CLI (`scripts/benchmarks/architecture_sweep.py`) that records parquet catalogues for architecture sweeps, unblocking upcoming 4090 profiling sessions.

**Exit Criteria**

* Reports meet investor presentation quality with minimal manual editing and highlight profitability decisions for run extensions.
* Monitoring dashboard tracks latency, throughput, ROI drift, experiment health, feature drift, and regime shift warnings in real time.
* Guardrail violations trigger structured alerts and optional automatic de-risking actions tied to reinforcement policy updates.
* Analyst review artefacts feed back into offline RL datasets without requiring external services or API keys.

---

### Dependencies & Tooling

* Python ≥3.10, PyTorch 2.2+, PyTorch Lightning 2.2+.
* Optional integrations: Weights & Biases, Pandera, TA-Lib via the `ta` package.
* CI to include `python scripts/ci_compile.py --quiet`, targeted `pytest` suites, and linting once the codebase stabilises.

### Contribution Expectations

* Update `task_tracker.md` with links to experiments and PRs when completing milestones.
* Keep documentation (`docs/`) in sync with new capabilities to reduce onboarding friction.
* Follow the structured logging patterns established in `market_nn_plus_ultra.utils.logging`.

