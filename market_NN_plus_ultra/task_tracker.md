# Market NN Plus Ultra — Task Tracker

This tracker organises the roadmap toward a production-ready "ultimate trader". Tasks are grouped by stage and prioritised for compounding ROI.

- [x] Finalise SQLite schema contract for `assets`, `series`, `indicators`, `trades`, and `benchmarks` tables (see `docs/sqlite_schema.md`).
- [x] Implement feature registry with pluggable indicator computation (TA-Lib, spectral, macro, alternative data).
- [x] Add data validation suite (Great Expectations or pandera) to guarantee clean inputs.
- [x] Define baseline experiment YAML files for different asset classes (equities, crypto, forex).
- [x] Document canonical data flow in `docs/architecture.md` (extend with schema diagrams).
- [x] Automate feature registry documentation export (Markdown generated from `FeaturePipeline.describe()`).
- [x] Ship Lightning training and data modules with YAML-driven configuration loading.

### Immediate Implementation Plan (Sprint 0)
- [x] Harden SQLite ingestion with schema validation + null/duplicate checks.
- [x] Implement end-to-end smoke test that runs the Plus Ultra agent against a sample SQLite fixture.
- [x] Wire evaluation metrics into the agent CLI output for faster iteration loops.
- [x] Expose rich CLI overrides for training and pretraining scripts so researchers can tweak depth, devices, and optimisation knobs without editing YAML files.
- [x] Profile omni-scale backbone throughput and memory on GPU/CPU to tune default hyperparameters.
- [x] Add CI job that exercises `python -m compileall` + unit stubs to protect against syntax drift. (See `.github/workflows/market-nn-plus-ultra-ci.yml`.)
  - [x] Ship `scripts/ci_compile.py` as a reusable compileall smoke test for local + CI pipelines.
- [x] Publish guidance for fusing long price histories, technicals, and alternative data into the SQLite fixtures so GPU runs see richer regimes. (Documented in `docs/sqlite_schema.md`.)
- [x] Generate reproducible, Pandera-validated fixtures sized to saturate a 24 GB GPU during long training sessions. (Use `scripts/make_fixture.py`.)

### Feature Registry Enhancements
- [x] Auto-generate documentation from `FeatureRegistry.describe()` into Markdown.
- [x] Add alternative data connectors (on-chain metrics, macro calendars) into the registry. (See `market_nn_plus_ultra.data.alternative_data` and regression coverage in `tests/test_alternative_data.py`.)
- [x] Document signal coverage expansion that pairs alternative data with technical indicators for long-horizon experiments. — Status: Published pairing matrix and workflow guidance in `docs/signal_coverage.md`.
  - Notes (2024-02-25): Outline will now cross-reference schema enforcement progress so the documentation includes validation guarantees and benchmark expectations highlighted in the optimisation log.
  - Notes (2025-10-19): Drafted section headers covering regime usage, macro/technical recipes, and CLI toggles so the documentation update can land as soon as the new switches are implemented.
  - Notes (2025-10-25): Consolidated telemetry inputs (diagnostics parquet exports, cross-asset profiling metrics) to ensure the documentation highlights how alternative-data features interplay with calibration-aware heads and PPO warm starts.
  - Notes (2025-10-26): Logged calibration-head and simulator dependencies so the documentation can reference confidence telemetry and PPO warm-start workflows once implemented.
  - Notes (2025-10-27): Added pairing matrix, regime-aware workflow checklist, and CLI toggle coverage so research teams can enable fused bundles without bespoke notebooks.
- [x] Surface dependency errors with structured logging for experiment reproducibility. (PR TBD)
- [x] Seed registry with higher-moment statistics and spectral energy factors.

## 1. Deep Temporal Modelling
- [x] Implement TemporalFusionTransformer-style encoder with multi-horizon support.
- [x] Add multi-resolution attention block mixing dilated convolutions and transformer layers.
- [x] Integrate state-space (S4/SSM) module for long-context retention.
- [x] Ship mixture-of-experts expansion for scaling feed-forward capacity without quadratic attention cost.
- [ ] Benchmark omni-scale backbone versus hybrid baseline across multiple asset universes with 4090 VRAM profiling. — Notes: Harness scaffolding shipped via `scripts/benchmarks/architecture_sweep.py`; awaiting GPU allocation to run full comparisons.
  - Notes (2025-10-26): Queued telemetry exports (diagnostics, fill-rate metrics) and prepared fixture variants with/without cross-asset tensors to accelerate analysis once GPU time is secured.
  - Notes (2025-10-28): Drafted GPU scheduling matrix, enumerated regression checkpoints for incremental sweeps, and recorded CPU-only smoke-test fallback to keep configs exercised until dedicated hardware is available.
  - Notes (2025-10-29): Prioritised telemetry ingestion automation, lined up profiler trace storage budgets, and scheduled dry-run sweeps using reduced fixtures to validate harness changes before full GPU allocation.
  - Notes (2025-11-05): Captured benchmark configs under `configs/benchmarks/`, logged profiler placeholders in `docs/benchmarks/architecture_sweep.md`, and added regression coverage in `tests/benchmarks/test_architecture_sweep.py`; telemetry remains pending dedicated GPU allocation.
- [x] Pretrain on masked time-series reconstruction before supervised fine-tuning (`scripts/pretrain.py`).
- [ ] Benchmark pretraining checkpoints vs. scratch initialisation across asset classes. — Notes: Will run after schema enforcement lands to guarantee clean fixtures.
   - Notes (2025-10-26): Defined evaluation splits aligned with diagnostics outputs so calibration drift and gradient-noise comparisons remain reproducible.
   - Notes (2025-10-28): Synced checkpoint catalogue with latest fixtures, earmarked masked/contrastive pairs for comparative runs, and listed telemetry exports required for profitability dashboards.
   - Notes (2025-10-29): Reserved benchmarking slots following warm-start smoke tests, documented metric rollups shared with profitability reporting, and queued automation scripts to publish comparison summaries alongside diagnostics logs.
   - Notes (2025-11-07): Sequenced GPU allocation requests alongside upcoming PPO rollout profiling, recorded dependency on profitability reporting schema updates, and outlined fallback CPU smoke tests to keep configs exercised while waiting for hardware.
- [x] Extend pretraining tasks with contrastive (TS2Vec-style) objectives for regime discrimination.
- [x] Introduce curriculum over window sizes and horizons to stabilise very deep models. (See `CurriculumScheduler`)
- [ ] Automate architecture sweeps over depth, horizon, and dilation for omni-scale, MoE, transformer, and state-space backbones. — Notes: Planned to extend the benchmark harness CLI once baseline sweeps are in place.
   - Notes (2025-10-26): Drafted CLI flag matrix tying sweep dimensions to diagnostics sampling intervals and documented GPU memory heuristics from recent telemetry runs.
   - Notes (2025-10-28): Outlined experiment-tracker batching to surface partial results, captured requirement to persist profiler traces, and aligned outputs with upcoming reporting automation.
   - Notes (2025-10-29): Stubbed CLI flag validation tests, scheduled integration with telemetry catalogue exports, and earmarked automation hooks for pushing sweep results into the reporting pipeline.
   - Notes (2025-11-07): Prioritised sweep automation backlog for Sprint 5, pairing distributed rollout telemetry with architecture profiling harnesses and drafting regression fixtures to validate CLI flag combinations before enabling GPU jobs.
- [x] Add gradient-noise diagnostics and calibration drift alerts to the training loop telemetry. — Notes: `TrainingDiagnosticsCallback` now logs gradient-noise ratios and calibration drift with configurable thresholds plus regression coverage.
  - Notes (2025-10-19): Telemetry schema draft enumerates gradient noise, calibration drift, throughput, and data freshness metrics to ensure future instrumentation plugs into reporting without rework.
  - Notes (2025-10-23): Prototyping `TrainingDiagnosticsCallback` scaffolding with CLI toggles for opt-in telemetry (`--diagnostics-profile`, `--diagnostics-interval`) while drafting regression hooks that compare diagnostics across fixture runs.
  - Notes (2025-10-24): Landed CLI/YAML toggles, callback thresholds, and tests for diagnostics parsing plus metric emission so supervised runs expose stability telemetry by default.
- [x] Integrate calibration-aware (Dirichlet/quantile) heads for safe scaling to deeper models. — Notes: `CalibratedPolicyHead` + Lightning integration landed with config toggles and regression coverage.
  - Notes (2024-02-25): Calibration head design will inherit empirical priors collected during the optimisation plan resync; keeping dependency on stability tooling explicit.
  - Notes (2025-10-25): Tagged diagnostics callback outputs and cross-asset fill-rate telemetry as required inputs for calibration-head validation, sequenced prototype tasks (loss wrappers, config toggles, regression fixtures) for the next sprint.
  - Notes (2025-10-26): Drafted calibration sweep plan pairing Dirichlet temperature/quantile spacing with PPO warm-start checkpoints for reinforcement-aware validation.
  - Notes (2025-10-28): Locked `CalibrationHeadAdapter` API sketch, mapped config validation coverage, and tracked simulator cost-tracing dependency to align calibrated confidence intervals with PPO reward shaping.
  - Notes (2025-10-29): Scheduled prototype implementation pairing embedding heads with calibration adapters, defined synthetic datasets for deterministic expectation checks, and coordinated telemetry schema changes with reporting/monitoring milestones.
  - Notes (2025-10-30): Implemented `market_nn_plus_ultra.models.calibration.CalibratedPolicyHead`, surfaced calibrated payloads through `MarketLightningModule.latest_head_output`, parsed YAML `model.calibration` blocks, and added `tests/test_calibration_head.py` to guard quantile monotonicity + concentration positivity.

## 2. Trading Objective & Reinforcement Learning
- [x] Add differentiable PnL simulator with position sizing, transaction costs, and slippage.
- [x] Implement utility-aware losses (Sharpe, Sortino, drawdown penalties) in `models.losses`.
- [x] Wrap policy gradient (PPO/IMPALA) fine-tuning on top of the supervised forecaster.
- [x] Support batched scenario simulations to stress-test policies.
- [x] Explore calibration-aware heads (Dirichlet / quantile) for action confidence.
  - Notes (2025-10-26): Outlined PPO-compatible inference wrappers and captured simulator latency constraints for sampling action confidences online.
  - Notes (2025-10-28): Split implementation into head-training experiments piggybacking on supervised checkpoints and policy-evaluation harness updates emitting confidence telemetry for reporting.
  - Notes (2025-10-29): Drafted PPO evaluation checklist covering confidence calibration metrics, defined simulator integration tests for latency impact, and coordinated telemetry outputs with the operations playbook.
  - Notes (2025-10-30): Baseline calibration head now emits quantile intervals + Dirichlet confidence suitable for PPO warm starts; follow-up tasks will wire outputs into reinforcement evaluation harnesses.
- [x] Warm-start RL fine-tuning runs from the masked/contrastive pretraining checkpoints via CLI switches.
  - Notes (2025-10-25): Drafted warm-start experiment matrix and telemetry capture checklist (diagnostics aggregates, rollout stability traces) to align with PPO upgrade planning in the implementation log.
  - Notes (2025-10-26): Finalised CLI contract draft (`--warm-start-checkpoint`, `--warm-start-tuning`) and mapped regression fixtures combining diagnostics snapshots with rollout summaries.
  - Notes (2025-10-28): Seeded synthetic PPO fixtures referencing latest pretraining checkpoints, documented replay-buffer initialisation for deterministic comparisons, and recorded rollback plan for unstable warm starts.
  - Notes (2025-10-29): Scheduled smoke-test executions against reduced fixtures, aligned checkpoint catalogue updates with reporting automation, and captured alerting requirements for failed warm starts in the monitoring milestone draft.
  - Notes (2025-11-04): Implemented warm-start CLI gating with resume semantics, executed the first diagnostics-enabled smoke run, and staged distributed rollout harness benchmarks for the next Sprint 4 review.
  - Notes (2025-11-05): Finalised backbone weight loading from masked/contrastive checkpoints via the PPO runner, added `--pretrain-checkpoint` validation to the CLI, and extended regression coverage to assert weights transfer before PPO updates.
- [ ] Extend PPO-style upgrades to optimise ROI directly using the differentiable PnL simulator after supervised convergence.
   - Notes (2024-02-25): PPO upgrades will reuse optimisation telemetry (latency, gradient noise) once Phase 2 diagnostics are instrumented, keeping action-confidence work grounded in measurable improvements.
   - Notes (2025-10-26): Sequenced simulator integration milestones (slippage hooks, latency buckets) with PPO reward-shaping toggles to align ROI optimisation with execution realism.
   - Notes (2025-10-28): Defined convergence gates (calibration drift threshold, drawdown guardrails) for enabling ROI-centric updates and queued benchmarking harness updates for ROI vs. risk comparisons.
   - Notes (2025-10-29): Planned incremental rollout of ROI-tuned objectives following warm-start smoke tests, allocated profiling windows for simulator latency validation, and aligned KPI reporting with profitability automation tasks.
   - Notes (2025-11-07): Locked Sprint 5 spike to map PPO reward-shaping hooks onto the upcoming simulator telemetry schema, drafted replay-buffer instrumentation requirements, and catalogued regression scenarios exercising risk-aware losses alongside ROI objectives.

## 3. Evaluation & Monitoring
- [x] Build evaluation harness for daily/weekly backtests with walk-forward splits.
- [x] Compute ROI, max drawdown, Calmar, hit rate, and tail risk metrics by default.
- [x] Ship inference agent + CLI to produce prediction frames from SQLite sources.
- [x] Generate Markdown/HTML reports with charts for quick inspection.
- [x] Wire up experiment tracking (Weights & Biases or MLflow) for metadata and artefacts.
- [x] Add automated guardrail metrics (exposure, turnover, tail percentiles) for live trading readiness.
- [ ] Publish automated run reports referencing the research agenda milestones. — Notes: Will piggyback on Phase 4 reporting upgrades after telemetry contracts stabilise.
  - Notes (2025-10-26): Logged telemetry exports (diagnostics aggregates, calibration sweeps, simulator cost breakdowns) that the reporting automation must ingest.
  - Notes (2025-10-28): Drafted milestone-referenced narrative stubs so automation can link experiment artefacts back to roadmap status without manual editing.
- [ ] Automate profitability summaries (ROI, Sharpe, drawdown) for every long training session. — Notes: Targeting the same reporting stack as Phase 4 Milestone 1.
   - Notes (2024-02-25): Report automation will ingest optimisation KPIs defined in the implementation log so profitability summaries capture calibration and stability diagnostics alongside ROI.
   - Notes (2025-10-26): Captured integration tasks for incorporating simulator-derived execution costs into summary tables and queued fixtures for validating drawdown aggregation under new telemetry fields.
   - Notes (2025-10-28): Added experiment-tracker upload cadence for interim profitability snapshots and documented schema extensions for calibration-derived confidence intervals.
   - Notes (2025-11-07): Coordinated telemetry schema updates with simulator and PPO teams, outlined Markdown/HTML template deltas for risk-aware metrics, and scheduled Sprint 5 doc automation spikes to keep reporting aligned with new objectives.

## 4. Automation & Deployment
- [ ] Containerise training + inference pipelines with GPU support. — Notes: Blocked until simulator + service interface prototypes settle.
  - Notes (2025-10-26): Drafted base-image requirements bundling diagnostics tooling and simulator dependencies to minimise integration friction once implementation begins.
  - Notes (2025-10-28): Selected candidate CUDA runtime baselines aligned with profiler expectations and listed smoke tests (dataset build, diagnostics callback, PPO rollout) containers must execute before publishing.
  - Notes (2025-10-29): Documented container layer caching strategy for telemetry artefacts, outlined GPU runtime validation scripts, and coordinated dependency pinning with upcoming service scaffolding.
- [ ] Expose REST/gRPC service that mirrors the existing market agent API but uses the new brain. — Notes: Capturing parity requirements while Phase 3 service scaffold is planned.
  - Notes (2025-10-26): Outlined telemetry payload schema aligning calibration heads, PPO warm starts, and simulator metrics for service contract readiness.
  - Notes (2025-10-28): Sketched streaming inference mode for multi-horizon forecasts, defined pagination contract for telemetry attachments, and logged authentication hooks required for deployment parity.
  - Notes (2025-10-29): Scheduled API review with monitoring/reporting leads, mapped response schema versioning to deployment automation, and prepared stub tests covering backward-compatible payloads.
- [ ] Schedule continuous retraining jobs triggered by new data arrival. — Notes: Will reuse orchestration DAG from Phase 3 Milestone 4 once prototyped.
  - Notes (2025-10-26): Scheduled spike for diagnostics ingestion operator and defined retry semantics for warm-start stages.
  - Notes (2025-10-28): Added change-data-capture polling requirements, calibration-drift alert thresholds, and storage budgets for rolling checkpoint windows.
  - Notes (2025-10-29): Broke DAG implementation into schema-validation, warm-start, PPO, and reporting tasks; queued integration tests for checkpoint lineage tracking; and aligned orchestration alerts with monitoring escalation paths.
  - Notes (2025-11-04): Drafted DAG task skeletons for simulator smoke tests and profitability report generation, added retry/idempotency guidance, and planned fixture-backed regression coverage for the dataset refresh → diagnostics export path.
- [ ] Set up online monitoring for live performance and drift detection. — Notes: Pending telemetry surface defined in Phase 4 Milestone 2.
  - Notes (2025-10-26): Matched diagnostics sampling cadence with Prometheus scrape intervals and scoped simulator latency histograms for alerting.
  - Notes (2025-10-28): Specified dashboard panels (calibration confidence, ROI trend, drawdown guardrails), alert routing rules, and dependencies on the service telemetry payload contract.
  - Notes (2025-10-29): Drafted alert playbooks with escalation timing, synchronised metric namespaces with service planning, and catalogued synthetic drift scenarios for end-to-end monitoring tests.
  - Notes (2025-11-04): Locked profitability reporting payload schema IDs with the service planning thread, reserved telemetry pagination retry semantics, and ensured monitoring dashboards can ingest the new diagnostics aggregates without schema churn.
- [ ] Build playbook for human-in-the-loop overrides and risk manager approvals. — Notes: Drafting outline alongside analyst feedback tooling requirements.
  - Notes (2025-10-26): Added placeholders for calibration-confidence dashboards and simulator what-if analyses to inform override decisions.
  - Notes (2025-10-28): Defined escalation tree integrating monitoring alerts with analyst annotations, added rollback SOP referencing checkpoint lineage, and documented review cadence expectations.
  - Notes (2025-10-29): Outlined approval runbooks tied to guardrail violations, mapped decision logging to analyst feedback tooling, and scheduled doc reviews alongside monitoring/service milestones.
  - Notes (2025-11-04): Captured telemetry ownership assignments (diagnostics parquet, simulator traces) so override workflows can surface provenance and profitability context during reviews.
- [ ] Integrate live monitoring, automated reporting, and risk guardrails into a single operations playbook for extend/branch decisions. — Notes: Will consolidate once reporting, monitoring, and guardrail milestones reach MVP.
  - Notes (2024-02-25): Operations playbook will surface optimisation KPIs (latency, calibration, guardrail triggers) defined in the implementation plan so deployment readiness decisions remain data-driven.
  - Notes (2025-10-26): Synced reporting and monitoring schema drafts to ensure playbook embeds linked dashboards and playback logs without refactors.
  - Notes (2025-10-28): Outlined chapter structure tying each milestone to operational runbooks so continuous retraining outputs feed directly into decision-support tooling when milestones land.
  - Notes (2025-10-29): Identified content owners for each playbook section, drafted cross-references to service and reporting updates, and scheduled validation workshops once MVP dashboards are available.

### Active Work Log — 2024-02-24

* Schema enforcement (Phase 1 Milestone 2) kicked off — Pandera models under design for `assets`, `series`, and `indicators` tables prior to CLI integration.
* Market-regime labelling (Phase 1 Milestone 4) discovery underway — collecting volatility/liquidity heuristics and alternative data touchpoints.
* Cross-asset feature view planning (Phase 1 Milestone 5) started — assessing ETF sector panel alignment strategies for SQLite-friendly joins.

### Active Work Log — 2025-10-17

* Market-regime labelling pipeline implemented with volatility/liquidity/rotation buckets and wired into synthetic fixture generation.
* Added regression coverage in `tests/test_labelling.py` to guarantee deterministic outputs and fallback behaviour when turnover is absent.
* Documented progress in the implementation log to unblock CLI wiring and benchmarking follow-ups for regime-aware experiments.

### Active Work Log — 2025-10-19

* Finalising CLI surface and validation matrix for regime labelling toggles so implementation can proceed without re-triage.
* Specifying cross-asset profiling instrumentation (latency, memory, alignment diagnostics) and identifying representative fixtures for benchmarking.
* Drafting documentation structure for combined signal coverage, tying sections to upcoming CLI toggles and validation references.
* Outlining telemetry schema (gradient noise, calibration drift, throughput) that Phase 2 diagnostics and reporting milestones will consume.

### Active Work Log — 2025-10-22

* Completed cross-asset alignment implementation, wiring the dataset-build CLI toggle, validation schema, and documentation updates into the release plan.
* Added regression coverage for the cross-asset view builder and CLI entry point to guarantee deterministic fills and feature naming.
* Published the cross-asset profiling script and recorded new telemetry fields (fill rate, dropped rows/features) for upcoming benchmarking runs.

### Active Work Log — 2025-10-23

* Initiated stability diagnostics implementation by defining callback lifecycles for gradient-noise estimation and calibration drift monitoring.
* Mapped CLI integration work for opt-in diagnostics, ensuring researchers can trigger telemetry without editing experiment YAML.
* Scheduled telemetry schema reviews so the new diagnostics artefacts feed Phase 2 reporting and Phase 4 monitoring milestones without rework.

### Active Work Log — 2025-11-03

* Ran Sprint 4 integration sync to coordinate calibration-head export wiring, PPO warm-start smoke tests, simulator profiling harness tasks, and reporting/monitoring dependencies.
* Logged regression/CI coverage gaps for PPO warm starts, simulator trace exports, and profitability report templates so upcoming PRs remain incremental.
* Sequenced documentation updates (`docs/sqlite_schema.md`, telemetry guide, operations playbook) with planned implementations to keep stakeholder onboarding smooth.
* Captured telemetry contract owners across diagnostics parquet exports, simulator trace payloads, and service response schemas to reduce cross-stream coordination overhead during implementation.

### Active Work Log — 2025-11-04

* Completed warm-start CLI gating, executed the PPO smoke run with calibration telemetry enabled, and staged distributed rollout benchmarks for the next Sprint 4 review.
* Ran deterministic simulator kernel smoke tests, capturing latency bucket histograms and validating telemetry payload schemas ahead of PPO integration.
* Locked profitability reporting payload contracts and propagated schema identifiers to service/orchestration planning to keep monitoring dashboards in sync.
* Sketched orchestration DAG skeletons for simulator smoke tests and profitability report generation with retry/idempotency notes and planned fixture-backed regression coverage.

### Active Work Log — 2025-10-24

* Shipped the production diagnostics callback with gradient-noise ratio tracking, calibration drift summaries, and thresholded warnings tied to config toggles.
* Added regression coverage for diagnostics statistics and YAML parsing so telemetry remains trustworthy as instrumentation expands.
* Updated default experiment configs and training CLI overrides to surface diagnostics controls without manual YAML edits.

### Active Work Log — 2025-10-26

* Coordinated calibration-head planning with PPO warm-start and simulator requirements, producing parameter sweep outlines and CLI contract drafts.
* Collected diagnostics exports to seed upcoming benchmarking tables and defined telemetry ingestion operator scope for orchestration work.
* Synced reporting templates, monitoring alerts, and service payload drafts so Sprint 4 implementation can start with aligned telemetry schemas.

### Active Work Log — 2025-10-28

* Broke Sprint 4 backlog into implementable slices covering calibration heads, PPO warm starts, profitability reporting, and monitoring dashboards.
* Registered benchmarking artefact requirements (diagnostics exports, simulator traces, cross-asset tensors) so architecture sweeps and checkpoint comparisons can run once GPU allocations land.
* Expanded orchestration and service design notes with change-data-capture polling, calibration-drift alerting, and escalation workflows to reduce re-triage when coding begins.

---

### Contributing Guidelines
* Keep tasks small and outcome-driven.
* Update statuses during each iteration.
* Link to PRs / experiments when tasks are completed.
