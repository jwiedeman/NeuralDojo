# Implementation Plan

This living plan translates the Market NN Plus Ultra roadmap into concrete engineering milestones. It is intended to complement `task_tracker.md` by providing narrative context, sequencing, and success criteria for the next stages of development.

## Optimization Execution Log

* **2024-02-25 — Plan resync:** Reconfirmed Phase 1 as the primary optimisation bottleneck. The focus is to finish schema enforcement and unlock richer market-regime labels before scaling benchmarking sweeps.
* **2024-02-25 — Traceability update:** Added implementation notes directly to each milestone below so in-flight work, blockers, and measurement hooks are visible without cross-referencing other docs.
* **2024-03-05 — Sprint 1 focus:** Locked scope for the next iteration: land Pandera schema enforcement, prepare regime labelling prototypes, and draft stability tooling scaffolds so Phase 2 experiments inherit validated datasets and diagnostics.
* **2024-03-08 — Schema validator landing:** Pandera-backed validation bundle merged into `market_nn_plus_ultra.data.validation`, unlocking CLI wiring work and surfacing structured logging hooks for downstream telemetry.
* **Next checkpoint:** Capture concrete telemetry requirements for stability tooling once schema validation lands (target: next iteration review).
* **2025-10-19 — Sprint 2 alignment:** Locked the follow-up slice of work to (a) wire regime-label toggles through the CLI and document troubleshooting, (b) prototype cross-asset join profiling with reproducible benchmarks, (c) catalogue combined alternative-data + technical signals for documentation, and (d) specify the telemetry schema (gradient noise, calibration drift, throughput) required for Phase 2 diagnostics. Progress will be recorded directly against the four active milestones below so downstream implementers can pick up remaining sub-tasks without re-triage.
* **2025-10-18 — Optimisation cadence reset:** Locked the follow-up sprint around wiring regime labelling into the CLI, staging stability diagnostics scaffolding, and cataloguing cross-asset join benchmarks so benchmarking runs inherit complete telemetry. Added explicit acceptance criteria for each outstanding milestone to keep dependency ordering visible while implementation proceeds.
* **2025-10-17 — Regime labelling pipeline:** Added deterministic volatility/liquidity/rotation labellers in `market_nn_plus_ultra.data.labelling`, wiring them into fixture generation and shipping regression tests for reproducibility.

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
2. **Schema enforcement** — Extend `market_nn_plus_ultra.data.validation` with pandera models for each joined table and wire the checks into the CLI tooling. Failure criteria should immediately halt training and emit structured logs, ensuring clean, fused datasets before scaling parameters. **Status:** ✅ Completed — `validation.py` now ships Pandera schemas for assets, price series, indicators, regimes, trades, and benchmarks alongside duplicate/sort guards, foreign-key enforcement, and structured logging helpers.
   - *Post-landing follow-up (2024-03-08):* Finish wiring the validators through the dataset assembly CLI (`market_nn_plus_ultra.cli.dataset_build`), expose a `--strict-validation` toggle, and add regression fixtures plus CI hooks so `python scripts/ci_compile.py` + schema smoke tests run together. Document failure-handling playbooks in `docs/sqlite_schema.md` with examples from the structured logs.
   - *Next steps (2024-02-25):* ✅ Finalised cross-table foreign-key assertions, codified null/duplicate rules, and planned CLI smoke tests that fail fast when fixtures drift. This groundwork enabled the validator merge.
   - *Work breakdown (2024-03-05):* ✅ (1) Published Pandera models for `assets`/`series`/`indicators` with shared validators, (2) integrated validators into dataset assembly helpers, (3) staged regression fixture coverage for upcoming CI hooks, and (4) outlined failure-handling playbooks for `docs/sqlite_schema.md`.
3. **Fixture generation** — Ship scripts under `scripts/` that can synthesise realistic OHLCV panels with long histories into SQLite for smoke testing (`scripts/make_fixture.py`) and document the workflow in `docs/sqlite_schema.md`. Generate reproducible fixtures that saturate GPU training with high-variance signals. **Status:** ✅ Completed via `scripts/make_fixture.py` and the expanded fusion guidance in `docs/sqlite_schema.md`. Notes captured in `docs/sqlite_schema.md#fixture-generation--data-fusion-guidance`.
   - *Notes (2024-02-25):* Queueing a refresh of the synthetic fixtures once schema enforcement is merged so benchmarking datasets reflect the stricter validation rules.
4. **Market-regime labelling** — Build deterministic pipelines in `market_nn_plus_ultra.data.labelling` that compute volatility regimes, liquidity regimes, and sector rotation markers using the enriched feature store so downstream models can condition on market state. **Status:** 🚧 In Progress — initial volatility/liquidity/rotation labellers shipped with Pandera-backed validation; follow-up work will expose CLI toggles and benchmarking sweeps.
   - *Implementation (2025-10-17):* Added `generate_regime_labels` with configurable quantile bands, integrated labelling into fixture generation, and backed the module with synthetic regression tests for determinism.
   - *Preparation (2024-02-25):* Drafting volatility band heuristics aligned with the optimisation focus on calibration heads; labelling design will piggyback on new Pandera contracts for reliability.
   - *Action items (2024-03-05):* ✅ (a) Translate heuristics into unit-tested labelling transforms once schema validators merge, ✅ (b) prototype volatility band parameter search notebooks referencing the optimisation metrics, and (c) line up synthetic fixture updates so labelling outputs land alongside schema enforcement — fixture wiring completed, parameter sweep notebooks still queued.
   - *Activation trigger (2024-03-08):* With schema enforcement merged, begin porting volatility/liquidity heuristics into `market_nn_plus_ultra.data.labelling` and schedule paired fixtures that exercise the new validation bundle.
   - *Next sprint goals (2025-10-18):* (1) Expose labelling toggles via `market_nn_plus_ultra.cli.dataset_build --regime-labels`, (2) extend regression fixtures to cover mixed-asset panels with label integrity checks, and (3) publish troubleshooting guidance in `docs/sqlite_schema.md` for mismatched label cardinality or stale quantile caches.
   - *In-flight planning (2025-10-19):* Finalising CLI surface design (`--regime-labels`, `--regime-bands`) and regression coverage that exercises single-asset, multi-asset, and stale-cache scenarios so documentation updates can include concrete error-handling playbooks once the toggles land.
5. **Cross-asset feature views** — Extend dataset assembly scripts to output aligned multi-ticker tensors (e.g., sector ETFs, index futures) that let the policy attend to correlations without requiring live data pulls. **Status:** 🗓 Planned — evaluating join strategies for synchronising ETF sector panels with the core ticker timelines while staying SQLite-friendly.
   - *Preparation (2024-02-25):* Documenting join performance benchmarks required for the optimisation sweep harness so we can measure feature-assembly cost versus training throughput.
   - *Preparation extension (2024-03-05):* Begin profiling candidate join strategies against regenerated fixtures, capture latency + memory metrics for the implementation log, and earmark config toggles so benchmarking sweeps can enable/disable cross-asset tensors cheaply.
    - *Profiling hook (2024-03-08):* Use the stricter validation outputs to seed multi-ticker fixture builds, recording join timings and memory in the optimisation log for next sprint planning.
    - *Benchmark checklist (2025-10-18):* (a) Collect baseline join timings for equities + ETF baskets on the refreshed fixtures, (b) document acceptable latency/regression thresholds for inclusion in benchmarking sweeps, and (c) stage a `--cross-asset-view` toggle in the CLI with a guardrail that enforces aligned calendars before merging.
   - *Instrumentation plan (2025-10-19):* Defining profiling hooks (memory snapshots, wall-clock timings, alignment-violation counts) and drafting fixture variations (sector ETF + equity basket, crypto + funding rates) so the benchmarking sweep captures stress cases before the CLI toggle ships.

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
   - *Scaffolding plan (2024-03-05):* Draft `TrainingDiagnosticsCallback` with pluggable collectors, outline gradient-noise estimation utilities, and sketch telemetry schema so outputs flow into Phase 4 reporting without refactors.
   - *Sprint 1 instrumentation (2025-10-18):* (i) Ship a thin callback shell that records batch-level noise scale estimates to parquet, (ii) wire calibration drift probes against existing validation splits, and (iii) log CLI warnings when diagnostics exceed thresholds defined in the optimisation log.
3. **Calibration head** — Introduce a Dirichlet/quantile hybrid head in `market_nn_plus_ultra.models` with supporting loss terms, enabling probability-calibrated signals for downstream risk controls and readying models for reinforcement fine-tuning. **Status:** 🗓 Planned — surveying prior experiments to determine default concentration priors before coding the head.
   - *Preparation (2024-02-25):* Collecting empirical priors from current supervised checkpoints to parameterise the calibration head once optimisation diagnostics are ready.
   - *Modelling notes (2024-03-05):* Catalogue supervised checkpoint metrics to anchor default Dirichlet concentration, design YAML config knobs for calibration modes, and plan unit tests that compare calibration error pre/post head integration.
   - *Implementation guardrails (2025-10-18):* Define acceptance tests covering (1) calibration error under synthetic drift scenarios, (2) compatibility with PPO fine-tuning checkpoints, and (3) fallback behaviour that reverts to deterministic heads when calibration diagnostics fail.
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
   - *Warm-start design (2025-10-18):* Establish CLI contract for `--warm-start-checkpoint` and `--curriculum-profile`, define regression tests that compare PPO convergence with/without warm starts on synthetic fixtures, and document failure cases in the optimisation log.
2. **Execution simulator** — Implement a vectorised trading simulator under `market_nn_plus_ultra.simulation` that models slippage, partial fills, funding costs, latency buckets, and capital constraints using historical limit-order approximations stored locally. **Status:** 🗓 Planned — compiling execution constraints gathered from prior NeuralArena deployments for simulator parameter defaults.
   - *Preparation (2024-02-25):* Mapping simulator telemetry outputs to the reporting stack to keep optimisation and deployment KPIs aligned.
   - *Prototype scope (2025-10-18):* Identify minimum viable order book fields to support latency buckets, craft deterministic fixtures for regression tests, and outline integration points with the differentiable PnL simulator before introducing stochastic fills.
3. **Risk-aware objectives** — Add composite Sharpe/Sortino/drawdown/CVaR losses to the reinforcement training loop with configuration flags to tune their weights, ensuring compatibility with supervised warm starts. **Status:** 🗓 Planned — identifying reusable loss components shared with supervised objectives to avoid duplicate code.
   - *Next steps (2024-02-25):* Reuse optimisation-ready loss components from supervised training to avoid divergence between ROI metrics across regimes.
   - *Validation plan (2025-10-18):* Create synthetic performance traces to benchmark gradient stability when combining ROI and CVaR losses, and document acceptable variance thresholds before enabling the objective bundle by default.
4. **Continuous retraining** — Build a Dagster/Airflow-compatible orchestration script under `scripts/automation/` that ingests fresh SQLite dumps, runs pretraining warm starts, supervised tuning, reinforcement fine-tuning, and backtests before pushing checkpoints to object storage. **Status:** 🗓 Planned — sketching DAG boundaries and artefact contracts in anticipation of simulator completion.
   - *Notes (2024-02-25):* Identify schedule hooks where optimisation dashboards can plug into orchestration alerts for drift detection.
   - *Planning checklist (2024-03-05):* Document DAG task interfaces, outline artefact versioning scheme aligned with schema enforcement outputs, and add telemetry requirements (latency, calibration drift) so retraining runs export the same metrics as manual experiments.
   - *Pipeline MVP (2025-10-18):* Draft YAML-driven DAG configuration referencing validated fixtures, codify checkpoint promotion rules tied to reporting KPIs, and plan integration tests that simulate partial failures with resumable stages.
5. **Service interface** — Scaffold a FastAPI or gRPC inference service in `market_nn_plus_ultra.service` that mirrors the existing market agent contract, streams the richer Plus Ultra telemetry, and exposes curriculum configuration for reinforcement fine-tuning. **Status:** 🗓 Planned — pencilling API parity requirements with the current market agent before implementation.
   - *Preparation (2024-02-25):* Defining optimisation KPIs (latency, throughput, calibration) that must be exposed by the service for Phase 4 monitoring.
   - *Readiness checklist (2025-10-18):* Specify service-level objectives for response latency and telemetry freshness, sketch contract tests that validate parity with the legacy market agent, and map deployment observability hooks required for Phase 4 monitoring.

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
   - *Notes (2024-03-05):* Enumerate telemetry inputs produced by the forthcoming diagnostics callbacks, map them to report sections, and draft a sample outline so implementation can start as soon as stability tooling lands.
   - *Notes (2024-03-08):* Extend the outline to reference schema validation artefacts (per-table failure summaries, CLI strict-mode logs) so automated reports surface data-quality provenance alongside performance metrics.
   - *Implementation notes (2025-10-18):* (a) Define templating slots for calibration drift, gradient noise, and PPO warm-start diagnostics, (b) script fixture-backed snapshot tests for Markdown/HTML parity, and (c) capture performance budgets to keep report generation under five minutes for 24 GB runs.
2. **Live monitoring** — Integrate Prometheus metrics and alerting hooks into the inference service. Document SLOs and alert policies in `docs/operations.md`, including live decision support for extending or branching experiments. **Status:** 🗓 Planned — deferring until the service scaffold from Phase 3 establishes telemetry contracts.
   - *Observability draft (2025-10-18):* Record required Prometheus exporters, define minimum alert set (latency, calibration drift, guardrail breaches), and align metric naming with the optimisation log for traceability.
3. **Risk guardrails** — Build guardrail modules that enforce exposure, turnover, tail-risk thresholds, and sector/factor caps during inference, configurable via YAML, and aligned with reinforcement fine-tuning outputs. **Status:** 🗓 Planned — aligning guardrail thresholds with upcoming calibration head outputs.
   - *Design hooks (2025-10-18):* Enumerate guardrail evaluation order, map fallback actions when violations occur, and earmark simulation scenarios to regression-test guardrail logic before production rollout.
4. **Analyst feedback loop** — Add review tooling in `market_nn_plus_ultra.reporting` that lets human supervisors annotate trades, flag anomalies, and record veto rationales stored in SQLite for future offline RL. **Status:** 🗓 Planned — capturing analyst workflow requirements while reinforcement hooks are designed.
   - *Workflow blueprint (2025-10-18):* Detail annotation schema extensions, draft UX flow for CLI-based annotations, and specify how feedback integrates with the reinforcement replay buffer.

### Progress Notes — 2024-03-08

* Landed the Pandera-backed schema validation bundle in `market_nn_plus_ultra.data.validation`, covering assets, prices, indicators, regimes, trades, and benchmarks with structured logging + foreign-key enforcement.
* Spun up follow-up tasks to wire validation into CLI assembly commands, document strict-mode usage, and capture regression fixtures for CI smoke tests.
* Triggered market-regime labelling implementation and cross-asset profiling prep now that validation guarantees are in place; scheduled fixture refreshes that exercise the new guards.

### Progress Notes — 2025-10-17

* Delivered the first deterministic regime labelling pipeline in `market_nn_plus_ultra.data.labelling`, covering volatility, liquidity, and rotation markers with quantile-driven heuristics.
* Hooked the labelling outputs into fixture generation and validation, ensuring synthetic SQLite bundles now emit multi-dimensional regime context by default.
* Added regression tests exercising the labelling heuristics against synthetic panels to guarantee determinism before wiring into benchmarking/CLI workflows.

### Progress Notes — 2025-10-19

* Scoped CLI toggles for regime labelling (`--regime-labels`, `--regime-bands`) alongside regression coverage for mixed-asset fixtures to unblock documentation of troubleshooting flows.
* Outlined cross-asset profiling harness requirements (memory, wall-clock, alignment diagnostics) and enumerated representative fixtures so benchmark data can be captured consistently in the optimisation log.
* Drafted documentation structure for combined alternative-data + technical signals, ensuring the upcoming write-up references the new CLI toggles and validation hooks.
* Itemised telemetry schema needs (gradient noise, calibration drift, throughput) so stability tooling and reporting milestones inherit a consistent metric contract.

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

