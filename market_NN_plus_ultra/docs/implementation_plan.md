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
* **2025-10-20 — Regime CLI wiring:** Delivered the dataset-build CLI with strict validation toggles, quantile overrides, regression coverage, and documentation updates so operational teams can regenerate labels without bespoke notebooks.
* **2025-10-21 — Sprint 3 kickoff:** Locked optimisation focus on landing cross-asset feature views, establishing profiling harnesses for the new joins, and translating telemetry requirements into concrete instrumentation tickets ahead of the Phase 2 simulator work. Sequenced deliverables so documentation, fixtures, and CLI toggles evolve together without blocking PPO upgrades.
* **2025-10-22 — Cross-asset alignment landing:** Implemented the alignment engine in `market_nn_plus_ultra.data.cross_asset`, wired the `--cross-asset-view` CLI flag with validation and structured logging, published the accompanying Pandera schema, and added the `scripts/benchmarks/cross_asset_profile.py` probe so profiling runs capture fill rates, dropped rows, and feature breadth for every dataset refresh.
* **2025-10-23 — Stability diagnostics instrumentation kickoff:** Sequenced the gradient-noise and calibration-drift callback prototypes, mapped CLI toggles for opt-in diagnostics runs, and scheduled telemetry schema reviews so reporting milestones can ingest the new metrics without refactors.
* **2025-10-25 — Sprint 3 checkpoint:** Sequenced the outstanding optimisation backlog around (a) deepening calibration-aware output heads, (b) wiring diagnostics artefacts into the reporting stack, and (c) preparing PPO warm-start experiments so the next iteration can move beyond planning. Captured dependency ordering across Phases 2–3 to reduce re-triage when implementation begins.
* **2025-10-24 — Diagnostics callback landing:** Shipped the production `TrainingDiagnosticsCallback` with gradient-noise ratio estimates, calibration drift summaries, CLI/YAML toggles, and regression coverage so stability telemetry is now part of every supervised run.
* **2025-10-27 — Signal coverage publication:** Promoted the alternative-data + technical pairing matrix to `docs/signal_coverage.md`, aligning CLI toggles, validation hooks, and regime-aware workflows so documentation unlocks the pending feature registry milestone.
* **2025-10-26 — Sprint 3 close-out & Sprint 4 kickoff prep:** Compiled telemetry from the new diagnostics callback, prioritised calibration-head research, and mapped the execution simulator + PPO warm-start contracts needed for the next optimisation slice. Sequenced doc/reporting updates with simulator milestones so Phase 3 remains unblocked once implementation starts.
* **2025-10-28 — Sprint 4 roadmap refinement:** Broke the pending calibration-head, PPO warm-start, and profitability reporting milestones into experiment-ready slices. Captured benchmarking artefact requirements (diagnostics exports, simulator traces, cross-asset tensors) so architecture sweeps and checkpoint comparisons can run in parallel once GPU time is secured. Logged automation hooks that the orchestration and monitoring milestones must expose to consume the upcoming telemetry bundle without rework.
* **2025-10-29 — Sprint 4 execution kickoff:** Locked day-one tasks around calibration-head prototyping, PPO warm-start smoke tests, and profitability reporting scaffolds. Sequenced shared dependencies (telemetry catalogues, simulator cost exports, config validation hooks) so each milestone can make progress without blocking others, and recorded coordination checkpoints with the monitoring and orchestration workstreams.
* **2025-10-30 — Calibration head release:** Landed the calibration-aware policy head, Lightning integration, YAML toggles, and regression coverage so telemetry, reporting, and PPO warm-start planning can consume calibrated confidence intervals without bespoke hooks.
* **2025-10-31 — Market-state embedding release:** Added categorical regime ingestion to the dataset loader, wired state-token windows through the DataModule, exposed YAML toggles, and shipped the `MarketStateEmbedding` module with regression coverage so policy backbones ingest volatility/liquidity context without manual feature engineering.
* **2025-11-01 — Sprint 4 mid-sprint sync:** Locked shared acceptance criteria for the PPO warm-start path, execution simulator MVP, and profitability reporting so the implementation threads can land without blocking each other. Captured integration touchpoints across CLI toggles, telemetry exporters, and regression fixtures directly below to keep the optimisation cadence predictable.
* **2025-11-02 — Telemetry alignment check:** Validated that calibration-head payloads, diagnostics callbacks, and forthcoming PPO curriculum hooks share consistent schema contracts. Logged follow-up instrumentation requirements for the simulator and reinforcement objectives so reporting/monitoring milestones ingest the richer metrics with no refactors.
* **2025-11-03 — Sprint 4 integration sync:** Sequenced the calibration-head export wiring, PPO warm-start smoke tests, and simulator profiling harness so downstream reporting and monitoring tasks inherit consistent artefacts. Captured remaining doc touchpoints and CI hooks needed before promoting the optimisation bundle to shared baselines.
* **2025-11-04 — Sprint 4 implementation focus:** Landed the warm-start CLI contract, ran first-pass simulator kernel smoke tests against deterministic fixtures, and locked profitability reporting telemetry payloads so the remaining reinforcement and automation milestones can ship without re-triage this iteration.
* **2025-11-05 — Warm-start backbone loader:** Finalised PPO warm-start support by loading masked/contrastive pretraining checkpoints directly into the policy backbone, wired dual CLI switches for supervised vs. self-supervised resumes, and extended the regression suite to guarantee the backbone state transfers without manual intervention.
* **2025-11-06 — Sprint 4 wrap & Sprint 5 scoping:** Locked remaining Sprint 4 close-out tasks (calibration export wiring, warm-start regression finalisation) and groomed Sprint 5 backlog slices around simulator kernel implementation, profitability reporting scaffolds, distributed PPO rollout profiling, and containerisation blueprints. Captured dependency handoffs between telemetry exporters, reporting templates, and orchestration hooks so follow-up PRs stay parallelisable.
* **2025-11-07 — Sprint 5 pre-kickoff alignment:** Sequenced simulator kernel work, risk-aware objective prototyping, and profitability automation so each thread can progress in parallel. Identified documentation and telemetry deltas required for the upcoming optimisation slice and recorded cross-team checkpoints to keep service and monitoring milestones unblocked.
* **2025-11-08 — Reporting CLI hardening:** Added a configurable charts directory override, expanded regression coverage for the reporting helpers, and validated Markdown/HTML parity through `tests/test_reporting.py` so profitability reporting scaffolds remain execution-ready for Sprint 5.
* **2025-11-09 — Execution simulator MVP:** Landed the vectorised execution engine in `market_nn_plus_ultra.simulation`, covering partial fills, latency penalties, funding costs, slippage, and position limits with dedicated regression tests so PPO and reporting workstreams can consume realistic execution telemetry.
* **2025-11-10 — Risk-aware reward shaping release:** Enabled configurable Sharpe/Sortino bonuses with drawdown and CVaR penalties via `RiskObjectiveConfig`, shipping reusable risk metric utilities and regression coverage so PPO runs inherit the new reward tuning knobs out of the box.
* **2025-11-10 — Research agenda reporting alignment:** Added milestone-aware report generation (`MilestoneReference`), CLI ingestion for agenda annotations, and regression coverage so automated reports now surface roadmap context by default.
* **2025-11-10 — Dilation sweep automation:** Extended the architecture benchmark CLI to accept multi-schedule dilation grids, persisted schedule identifiers in parquet outputs, and refreshed regression coverage so optimisation sweeps now traverse depth/horizon/dilation combinations without manual YAML edits.
* **2025-11-11 — Replay buffer PPO upgrade:** Delivered a CPU-backed replay buffer with configurable sampling ratios, sample-count reporting in the CLI summary, and regression coverage ensuring warm-started PPO runs mix fresh and historical experiences.
* **2025-11-12 — Operations readiness summary aggregator:** Added `compile_operations_summary` to bundle risk metrics, guardrail diagnostics, and threshold-driven alerts, unblocking the operations playbook milestones with production-ready payloads for approval workflows and dashboards.
* **2025-11-13 — Continuous retraining orchestrator:** Implemented the automation package under `market_nn_plus_ultra.automation` and the CLI entry point `scripts/automation/retrain.py`, wiring dataset validation, pretraining, supervised training, and PPO warm starts into a reproducible workflow with artifact exports for downstream reporting.
* **2025-11-14 — Distributed rollout worker pool:** Introduced a spawn-based rollout manager that fans out PPO data collection across CPU workers, added CLI overrides for worker counts/devices, and shipped regression coverage so sample totals scale with the configured worker pool while remaining replay-buffer compatible.
* **2025-11-15 — PPO risk & cost overrides:** Expanded `scripts/rl_finetune.py` to expose activation, trading-cost, and risk-objective overrides with regression coverage, ensuring reinforcement experiments can tune reward shaping and execution assumptions directly from the CLI.
* **2025-11-16 — PPO telemetry batching release:** Instrumented `market_nn_plus_ultra.training.reinforcement` with `RolloutTelemetry`, aggregated throughput/reward dispersion metrics across parallel rollout workers, refreshed the CLI summary, and expanded regression coverage so distributed profiling captures collection latency and sample rates alongside reward statistics.
* **2025-11-17 — PPO curriculum release:** Wired PPO fine-tuning to honour dataset curriculum schedules, record per-update curriculum metadata, and added a CLI `--curriculum-profile` inspector so researchers can review staged window/horizon changes before launching runs.
* **2025-11-18 — Multi-scale backbone release:** Delivered the hierarchical `MultiScaleBackbone` with configurable scale factors, fusion attention, training/pretraining integration, and regression coverage so architecture sweeps can quantify hierarchical context gains alongside omni/state-space baselines.
* **2025-11-18 — Reporting enrichment:** Completed the automated run-report upgrades with profitability summaries, attribution tables, scenario analysis, and bootstrapped confidence intervals, refreshing the CLI, documentation, and regression coverage so evaluation artefacts are investor-ready out of the box.
* **2025-11-18 — Inference service MVP:** Shipped the FastAPI service scaffold under `market_nn_plus_ultra.service` with prediction, configuration, curriculum, and reload endpoints, added regression coverage, and published the `scripts/service.py` launcher so the Plus Ultra agent can be served with telemetry-rich responses without bespoke notebooks.

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
4. **Market-regime labelling** — Build deterministic pipelines in `market_nn_plus_ultra.data.labelling` that compute volatility regimes, liquidity regimes, and sector rotation markers using the enriched feature store so downstream models can condition on market state. **Status:** ✅ Completed — deterministic labellers ship with CLI toggles, strict validation wiring, documentation updates, and regression coverage for multi-asset panels.
   - *Implementation (2025-10-17):* Added `generate_regime_labels` with configurable quantile bands, integrated labelling into fixture generation, and backed the module with synthetic regression tests for determinism.
   - *Preparation (2024-02-25):* Drafting volatility band heuristics aligned with the optimisation focus on calibration heads; labelling design will piggyback on new Pandera contracts for reliability.
   - *Action items (2024-03-05):* ✅ (a) Translate heuristics into unit-tested labelling transforms once schema validators merge, ✅ (b) prototype volatility band parameter search notebooks referencing the optimisation metrics, and (c) line up synthetic fixture updates so labelling outputs land alongside schema enforcement — fixture wiring completed, parameter sweep notebooks still queued.
   - *Activation trigger (2024-03-08):* With schema enforcement merged, begin porting volatility/liquidity heuristics into `market_nn_plus_ultra.data.labelling` and schedule paired fixtures that exercise the new validation bundle.
   - *Next sprint goals (2025-10-18):* ✅ (1) Expose labelling toggles via `market_nn_plus_ultra.cli.dataset_build --regime-labels`, ✅ (2) extend regression fixtures to cover mixed-asset panels with label integrity checks, and ✅ (3) publish troubleshooting guidance in `docs/sqlite_schema.md` for mismatched label cardinality or stale quantile caches.
   - *In-flight planning (2025-10-19):* ✅ Finalised CLI surface design (`--regime-labels`, `--regime-bands`) and regression coverage that exercises single-asset, multi-asset, and stale-cache scenarios so documentation updates can include concrete error-handling playbooks once the toggles land.
   - *CLI integration (2025-10-20):* Landed the dataset-build CLI with quantile overrides, strict validation toggles, regression-backed multi-asset fixtures, and README/`docs/sqlite_schema.md` troubleshooting playbooks.
5. **Cross-asset feature views** — Extend dataset assembly scripts to output aligned multi-ticker tensors (e.g., sector ETFs, index futures) that let the policy attend to correlations without requiring live data pulls. **Status:** ✅ Completed — cross-asset alignment now lives in `market_nn_plus_ultra.data.cross_asset`, integrates with the dataset-build CLI, and ships with regression coverage plus documentation updates describing the new SQLite table and troubleshooting flow.
   - *Preparation (2024-02-25):* Documenting join performance benchmarks required for the optimisation sweep harness so we can measure feature-assembly cost versus training throughput.
   - *Preparation extension (2024-03-05):* Begin profiling candidate join strategies against regenerated fixtures, capture latency + memory metrics for the implementation log, and earmark config toggles so benchmarking sweeps can enable/disable cross-asset tensors cheaply.
    - *Profiling hook (2024-03-08):* Use the stricter validation outputs to seed multi-ticker fixture builds, recording join timings and memory in the optimisation log for next sprint planning.
    - *Benchmark checklist (2025-10-18):* (a) Collect baseline join timings for equities + ETF baskets on the refreshed fixtures, (b) document acceptable latency/regression thresholds for inclusion in benchmarking sweeps, and (c) stage a `--cross-asset-view` toggle in the CLI with a guardrail that enforces aligned calendars before merging.
   - *Optimisation sync (2025-10-25):* Tagged the alignment engine as a dependency for PPO rollout profiling and calibration-head validation so telemetry captures fill-rate context alongside regime labels during the next sprint.
   - *Instrumentation plan (2025-10-19):* Defining profiling hooks (memory snapshots, wall-clock timings, alignment-violation counts) and drafting fixture variations (sector ETF + equity basket, crypto + funding rates) so the benchmarking sweep captures stress cases before the CLI toggle ships.
   - *Implementation breakdown (2025-10-21):* (1) Finalise schema contracts for joined cross-asset tensors using the existing Pandera validators as templates, (2) prototype the alignment engine inside `market_nn_plus_ultra.data.feature_pipeline` with synthetic fixtures covering mixed frequencies, (3) wire the `--cross-asset-view` toggle through `market_nn_plus_ultra.cli.dataset_build` with validation-aware error messaging, (4) capture benchmark scripts under `scripts/benchmarks/` that emit wall-clock + peak-memory telemetry, and (5) schedule documentation updates in `docs/sqlite_schema.md` describing alignment failure remediation.
   - *Completion (2025-10-22):* Delivered `market_nn_plus_ultra.data.cross_asset` with fill-rate statistics, persisted the new `cross_asset_views` table via the CLI flag (including strict Pandera validation), added `scripts/benchmarks/cross_asset_profile.py` for reproducible profiling, and expanded `docs/sqlite_schema.md` with usage examples plus troubleshooting guidance covering fill limits and dropped rows.

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
2. **Stability tooling** — Implement gradient noise scale diagnostics, calibration drift monitoring, and loss landscape plots within `market_nn_plus_ultra.training.train_loop`. Persist diagnostics to disk for every experiment and surface warnings in the CLI. **Status:** ✅ Completed — Gradient-noise and calibration telemetry now ship via `TrainingDiagnosticsCallback` with configurable thresholds, CLI toggles, and regression coverage.
   - *Next steps (2024-02-25):* Prototype callback scaffolding while awaiting validated datasets so optimisation metrics (noise scale, calibration drift) have trustworthy baselines.
   - *Scaffolding plan (2024-03-05):* Draft `TrainingDiagnosticsCallback` with pluggable collectors, outline gradient-noise estimation utilities, and sketch telemetry schema so outputs flow into Phase 4 reporting without refactors.
   - *Sprint 1 instrumentation (2025-10-18):* (i) Ship a thin callback shell that records batch-level noise scale estimates to parquet, (ii) wire calibration drift probes against existing validation splits, and (iii) log CLI warnings when diagnostics exceed thresholds defined in the optimisation log.
   - *Implementation kickoff (2025-10-23):* Lock callback lifecycles and CLI toggles (`--diagnostics-profile`, `--diagnostics-interval`) while drafting regression hooks that compare telemetry outputs across fixture runs to guard against silent drift.
   - *Landing notes (2025-10-24):* Production callback now emits gradient-noise ratios, calibration drift statistics, and threshold-based warnings while respecting YAML/CLI overrides.
3. **Calibration head** — Introduce a Dirichlet/quantile hybrid head in `market_nn_plus_ultra.models` with supporting loss terms, enabling probability-calibrated signals for downstream risk controls and readying models for reinforcement fine-tuning. **Status:** ✅ Completed — shipped the `CalibratedPolicyHead` with monotonic quantile projections, Dirichlet confidence outputs, Lightning integration, config toggles, and regression tests covering parsing + telemetry capture.
   - *Preparation (2024-02-25):* Collecting empirical priors from current supervised checkpoints to parameterise the calibration head once optimisation diagnostics are ready.
   - *Modelling notes (2024-03-05):* Catalogue supervised checkpoint metrics to anchor default Dirichlet concentration, design YAML config knobs for calibration modes, and plan unit tests that compare calibration error pre/post head integration.
   - *Implementation guardrails (2025-10-18):* Define acceptance tests covering (1) calibration error under synthetic drift scenarios, (2) compatibility with PPO fine-tuning checkpoints, and (3) fallback behaviour that reverts to deterministic heads when calibration diagnostics fail.
   - *Integration prep (2025-10-25):* Linked diagnostics callback outputs, cross-asset fill-rate telemetry, and alternative-data coverage metrics into a shared validation harness outline so calibration experiments ship with reproducible reports and documentation hooks.
   - *Validation prep (2025-10-26):* Drafted parameter-sweep plan covering Dirichlet temperature and quantile spacing, pairing each configuration with PPO warm-start checkpoints to verify reinforcement compatibility during calibration experiments.
   - *API sketch (2025-10-28):* Finalised the `CalibrationHeadAdapter` interface, mapped config validation coverage, and logged dependency on simulator cost tracing so calibrated confidence intervals remain aligned with PPO reward shaping.
   - *Landing notes (2025-10-30):* Implemented `market_nn_plus_ultra.models.calibration.CalibratedPolicyHead`, wired Lightning modules to expose the latest calibration payloads, extended YAML parsing with `model.calibration` toggles, and added regression coverage in `tests/test_calibration_head.py` to guarantee monotonic quantiles, positive Dirichlet concentrations, and config round-trips.
4. **Multi-scale backbone experiments** — Add hierarchical Transformer and state-space variants in `market_nn_plus_ultra.models.backbones` that fuse intraday/daily/weekly encoders, reporting their contribution to Sharpe and drawdown metrics. **Status:** ✅ Completed — shipped the `MultiScaleBackbone` with configurable scale factors, fusion attention, CLI integration, and regression coverage so benchmarking harnesses can compare hierarchical context against existing omni/state-space baselines.
   - *Preparation (2024-02-25):* Aligning config templates with optimisation sweep requirements (depth, dilation, device budgeting) so experiments start with consistent measurement hooks.
   - *Profiling queue (2025-10-26):* Reserved GPU profiling slots and drafted telemetry capture checklist (gradient noise, calibration drift, throughput) so multi-scale trials align with the diagnostics cadence introduced in Sprint 3.
   - *Sweep scheduling (2025-10-28):* Outlined batching/queueing strategy using the experiment tracker to publish interim results and persisting profiler traces for downstream reporting automation.
   - *Landing notes (2025-11-18):* Implemented `market_nn_plus_ultra.models.multi_scale.MultiScaleBackbone`, wired architecture selection through the training and pretraining loops, exposed YAML knobs for scale factors and fusion heads, and added unit coverage to guarantee shape preservation on irregular sequence lengths.
5. **Market-state embeddings** — Train volatility and regime embedding heads that feed into the policy layers, ensuring the configuration system can toggle them for ablation studies. **Status:** ✅ Completed — categorical regime signals now flow through the dataset, DataModule, and Lightning module with configurable embedding heads and regression coverage.
   - *Notes (2024-02-25):* Waiting on market-regime labelling outputs to set embedding dimensionality and regularisation schedules that will feed optimisation metrics.
   - *Telemetry sync (2025-10-26):* Coordinated embedding metrics (regime coverage, volatility reconstruction loss) with diagnostics logging so future experiments surface embedding health alongside calibration stats.
   - *Kickoff tasks (2025-10-29):* Drafted embedding module API sketch aligned with calibration-head adapters, stubbed config toggles for embedding ablations, and scheduled fixture updates that include the new embedding tensors for upcoming PPO warm-start experiments.
   - *Implementation (2025-10-31):* Landed regime-token ingestion in `SQLiteMarketDataset`, extended `SlidingWindowDataset` and the DataModule with state-token windows, introduced `MarketStateEmbedding`, wired YAML toggles, and added unit coverage exercising the new embedding path.

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

1. **PPO upgrades** — Extend `scripts/rl_finetune.py` with distributed rollouts, curriculum schedules, reward shaping toggles, and hooks that warm-start from the masked/contrastive pretraining checkpoints. Add replay buffers for optional off-policy learning and entropic-risk penalties. **Status:** ✅ Completed — distributed rollouts, curriculum scheduling, warm-start toggles, replay buffers, risk shaping, and telemetry now ship with CLI profiling hooks.
   - *Notes (2024-02-25):* Drafting rollout profiling checklist so PPO upgrades inherit the optimisation metrics (latency, gradient noise) prepared in Phase 2.
   - *Warm-start design (2025-10-18):* Establish CLI contract for `--warm-start-checkpoint` and `--curriculum-profile`, define regression tests that compare PPO convergence with/without warm starts on synthetic fixtures, and document failure cases in the optimisation log.
   - *Profiling prep (2025-10-21):* Outline telemetry adapters that consume the upcoming cross-asset join benchmarks so curriculum schedulers can react to feature breadth, and queue fixture variants that stress-test distributed rollout throughput before implementation begins.
   - *Sprint 3 checkpoint (2025-10-25):* Locked experiment matrix covering warm-start baselines, diagnostics-enriched rollouts, and curriculum variants; enumerated telemetry exports (diagnostics aggregates, cross-asset fill metrics, Sharpe/drawdown snapshots) for integration with reporting and monitoring milestones.
   - *Distributed design (2025-10-26):* Finalised evaluator/learner RPC boundaries, telemetry batching cadence, and fault-tolerance expectations so the upcoming implementation sprint can start with a clear interface contract.
   - *Experiment staging (2025-10-28):* Seeded deterministic PPO fixtures referencing the latest pretraining checkpoints, documented replay-buffer initialisation for reproducible warm starts, and added rollback criteria if curriculum schedules destabilise early episodes.
   - *Implementation kickoff (2025-10-29):* Stubbed learner/evaluator scaffolding with placeholder interfaces, scheduled smoke tests for warm-start checkpoints using reduced fixture slices, and synced telemetry export schemas with the reporting milestone to avoid duplication.
   - *Mid-sprint progress (2025-11-01):* Implemented the CLI surface (`--warm-start-checkpoint`, `--ppo-diagnostics-profile`) with validation guarding incompatible curriculum selections, wired calibration-head payload capture into rollout summaries, and drafted regression notebooks comparing warm-start vs. scratch convergence on synthetic fixtures.
   - *Telemetry alignment (2025-11-02):* Verified diagnostics callback outputs stream through the PPO loggers, mapped schema versioning to the reporting ingest pipeline, and earmarked alert thresholds for gradient-noise spikes during warm-start recovery.
   - *Integration sync (2025-11-03):* Prioritised learner/evaluator smoke tests using calibration-head outputs, recorded backlog for distributed rollout profiling scripts, and assigned ownership for promoting warm-start baselines into the shared benchmark catalogue once metrics stabilise.
   - *Implementation progress (2025-11-04):* Landed CLI gating for warm-start checkpoints with resume semantics, executed first warm-start smoke run capturing diagnostics payloads, and prepped distributed rollout harness benchmarks for the next iteration review.
   - *Warm-start landing (2025-11-05):* Added backbone checkpoint loader shared across supervised and self-supervised runs, exposed `--pretrain-checkpoint`/`--warm-start-tuning` CLI switches, and shipped regression coverage ensuring PPO initialises from masked pretraining weights without extra scaffolding.
   - *Distributed rollout planning (2025-11-06):* Broke remaining implementation into learner/evaluator worker pools with structured telemetry batching, defined CLI/Config validation for population-based curriculum toggles, and mapped profiling harness checkpoints (latency, throughput, gradient-noise deltas) required before enabling large-scale runs.
   - *Sprint 5 prep (2025-11-07):* Finalised backlog split between distributed rollout RPC wiring, replay buffer benchmarking, and curriculum scheduling experiments; captured dependencies on simulator telemetry exports and profitability reporting payload changes ahead of implementation start.
   - *Replay buffer landing (2025-11-11):* Implemented the CPU-hosted replay buffer with configurable sampling ratios, surfaced sample counts in PPO summaries, and added regression coverage so off-policy batches are exercised alongside baseline rollouts.
   - *Distributed rollout release (2025-11-14):* Spawned rollout worker pool with CLI overrides for worker count/device, added regression coverage validating sample scaling, and documented replay-buffer compatibility so profiling automation can focus on curriculum + telemetry batching next.
   - *Curriculum release (2025-11-17):* Reinforcement loop now synchronises curriculum stages across the main process and rollout workers, records per-update schedule metadata, and exposes a CLI `--curriculum-profile` preview so staged window/horizon changes are transparent before PPO launches.
   - *Telemetry landing (2025-11-16):* Added `RolloutTelemetry` aggregation across workers, surfaced reward dispersion and throughput metrics in `ReinforcementUpdate`, refreshed CLI output, and extended regression tests so profiling harnesses inherit collection latency and sample-rate telemetry for distributed runs.
   - *CLI expansion (2025-11-15):* Surfaced activation, trading-cost, and risk-objective overrides in `scripts/rl_finetune.py` with unit coverage so PPO runs can sweep reward shaping parameters without editing experiment YAML files, aligning telemetry outputs with the optimisation log.
2. **Execution simulator** — Implement a vectorised trading simulator under `market_nn_plus_ultra.simulation` that models slippage, partial fills, funding costs, latency buckets, and capital constraints using historical limit-order approximations stored locally. **Status:** ✅ Completed — shipped `simulation.engine` with configurable slippage/latency/funding knobs, position caps, and regression coverage in `tests/test_simulation.py`.
   - *Preparation (2024-02-25):* Mapping simulator telemetry outputs to the reporting stack to keep optimisation and deployment KPIs aligned.
   - *Prototype scope (2025-10-18):* Identify minimum viable order book fields to support latency buckets, craft deterministic fixtures for regression tests, and outline integration points with the differentiable PnL simulator before introducing stochastic fills.
   - *Data dependency checklist (2025-10-21):* Confirm cross-asset feature tensors expose bid/ask depth surrogates required by the simulator, document required sampling frequencies, and reserve CLI hooks for exporting simulator-ready parquet bundles once the join pipeline lands.
   - *Interface sync (2025-10-26):* Drafted API sketch aligning simulator slippage hooks with PPO reward shaping inputs and earmarked latency budget benchmarks sourced from diagnostics telemetry.
   - *Kickoff notes (2025-10-29):* Assigned engineering spikes for vectorised fill kernels and latency bucket calibration, prepared fixture export scripts for deterministic smoke tests, and mapped simulator logging fields to the profitability reporting milestone.
   - *Progress check (2025-11-01):* Finalised fixture schema covering limit-order depth snapshots and funding costs, prototyped vectorised fill kernels with benchmark targets noted, and aligned slippage parameter defaults with calibration-head uncertainty outputs for downstream PPO reward shaping.
   - *Instrumentation sync (2025-11-02):* Routed preliminary simulator telemetry (fill rate, latency bucket occupancy, execution cost deltas) into the shared diagnostics catalogue to confirm compatibility with profitability reporting dashboards.
   - *Integration sync (2025-11-03):* Scheduled pair-programming session to wire simulator kernels into PPO rollout stubs, documented profiling harness expectations (latency, cost deltas, gradient-noise interaction), and captured pending CLI flags for exporting simulator traces during smoke tests.
   - *Implementation progress (2025-11-04):* Ran deterministic fill-kernel smoke tests against synthetic fixtures, instrumented latency bucket summaries for telemetry export, and queued profiling scripts for the simulator harness ahead of PPO integration work.
   - *Module scaffolding (2025-11-06):* Finalised `market_nn_plus_ultra.simulation` package layout (vectorised fill kernels, funding cost adapters, telemetry dataclasses), assigned ownership for fixture-backed regression suites, and documented CLI surface expectations (`--simulator-profile`, `--execution-trace-export`) ahead of first integration PRs.
   - *Sprint 5 prep (2025-11-07):* Ranked kernel implementation tasks (fill allocation, latency buckets, funding rollovers), scheduled profiling harness smoke tests aligned with upcoming PPO rollout work, and collected telemetry schema deltas required by reporting and monitoring milestones.
   - *Landing notes (2025-11-09):* Delivered the `simulate_execution` API with latency bucket penalties, funding cost accumulation, and position caps, plus telemetry-friendly outputs (slippage, fees, funding) so PPO reward shaping and reporting automation can wire against consistent execution traces.
3. **Risk-aware objectives** — Add composite Sharpe/Sortino/drawdown/CVaR losses to the reinforcement training loop with configuration flags to tune their weights, ensuring compatibility with supervised warm starts. **Status:** ✅ Completed — risk-aware reward shaping now ships via `RiskObjectiveConfig`, blending Sharpe/Sortino bonuses with drawdown/CVaR penalties and regression coverage.
   - *Next steps (2024-02-25):* Reuse optimisation-ready loss components from supervised training to avoid divergence between ROI metrics across regimes.
   - *Validation plan (2025-10-18):* Create synthetic performance traces to benchmark gradient stability when combining ROI and CVaR losses, and document acceptable variance thresholds before enabling the objective bundle by default.
   - *Sprint 5 scoping (2025-11-07):* Drafted implementation outline covering reusable loss adapters, curriculum-aware weighting schedules, and regression fixtures comparing Sharpe/Sortino/CVaR blends; flagged dependencies on simulator cost telemetry and PPO diagnostics exports to keep instrumentation aligned.
   - *Telemetry alignment (2025-10-21):* Map simulator output fields (slippage, drawdown envelopes) to loss-scheduling knobs so once the simulator prototype is ready the objective bundle can ingest telemetry without rework; capture placeholder metrics in the optimisation log to backfill reports during implementation.
   - *Scheduling heuristics (2025-10-26):* Outlined loss-weight annealing strategies tied to gradient-noise ratios and queued experiment templates comparing ROI vs. CVaR emphasis under identical simulator traces.
   - *Implementation prep (2025-10-29):* Drafted reusable loss wrapper interfaces shared with supervised objectives, earmarked integration tests combining Sharpe/CVaR penalties, and aligned configuration defaults with the calibration-head planning notes.
   - *Design workshop (2025-11-01):* Reviewed objective-weight annealing strategies against newly captured calibration-head uncertainty metrics, defined default CVaR clipping bounds, and mapped telemetry fields required for profitability dashboards.
   - *Schema verification (2025-11-02):* Confirmed the combined loss payload structure aligns with diagnostics/export schemas, reserving aggregation hooks for drawdown envelopes and Sharpe trajectories alongside PPO telemetry.
   - *Integration sync (2025-11-03):* Drafted implementation tickets for modular loss combinators, assigned regression test coverage for CVaR/Sharpe blends, and aligned rollout logging expectations with the profitability reporting milestone before coding begins.
   - *Implementation progress (2025-11-04):* Prototyped composite ROI/Sharpe/CVaR loss wrappers sharing state with supervised objectives, exercised gradient-stability checks on synthetic traces, and recorded follow-up tasks to align loss-weight schedules with simulator telemetry cadence.
   - *Tuning roadmap (2025-11-06):* Sequenced annealing experiments across simulator latency buckets, paired calibration-head uncertainty exports with CVaR clipping sweeps, and earmarked cross-validation harness updates so supervised + reinforcement losses stay numerically stable under mixed precision.
   - *Landing notes (2025-11-10):* Introduced `RiskObjectiveConfig`-driven reward shaping with per-step Sharpe/Sortino bonuses and drawdown/CVaR penalties, shipped reusable risk metric utilities, and added regression tests (`tests/test_risk_objectives.py`) covering both metrics and reward blending.
4. **Continuous retraining** — Build a Dagster/Airflow-compatible orchestration script under `scripts/automation/` that ingests fresh SQLite dumps, runs pretraining warm starts, supervised tuning, reinforcement fine-tuning, and backtests before pushing checkpoints to object storage. **Status:** ✅ Completed — delivered the `run_retraining_plan` automation module with a CLI wrapper so validation, training, and PPO stages execute sequentially with persisted checkpoints and policy exports.
   - *Notes (2024-02-25):* Identify schedule hooks where optimisation dashboards can plug into orchestration alerts for drift detection.
   - *Planning checklist (2024-03-05):* Document DAG task interfaces, outline artefact versioning scheme aligned with schema enforcement outputs, and add telemetry requirements (latency, calibration drift) so retraining runs export the same metrics as manual experiments.
   - *Pipeline MVP (2025-10-18):* Draft YAML-driven DAG configuration referencing validated fixtures, codify checkpoint promotion rules tied to reporting KPIs, and plan integration tests that simulate partial failures with resumable stages.
   - *Dependency tracking (2025-10-21):* Record the forthcoming cross-asset tensors and simulator artefacts as orchestration inputs, sketch stub operators for benchmarking + telemetry uploads, and identify storage quotas needed once PPO upgrades begin exporting distributed rollout logs.
   - *Telemetry ingestion spike (2025-10-26):* Scheduled prototype for a diagnostics ingestion operator, defining retry semantics for warm-start tasks and outlining metadata schemas for run-to-run comparisons.
   - *Automation hooks (2025-10-28):* Added change-data-capture polling requirements, calibration-drift alert thresholds, and storage budgets for rolling checkpoint windows to the orchestration design log.
   - *Mid-sprint sync (2025-11-01):* Validated that simulator telemetry and PPO diagnostics share artefact versioning identifiers, updated DAG drafts with warm-start checkpoint lineage stages, and pencilled regression tests for resume-from-failure scenarios covering simulator/PPO boundaries.
   - *Alerting alignment (2025-11-02):* Drafted Prometheus-compatible metrics exporters for orchestration runs (latency buckets, calibration drift, PPO convergence flags) and scheduled pairing with monitoring milestones once simulator MVP lands.
   - *Integration sync (2025-11-03):* Broke orchestration MVP into promotable subgraphs (dataset refresh, diagnostics export, PPO warm start, profitability reporting), captured CI smoke-test requirements, and logged dependencies on upcoming simulator CLI hooks.
   - *Implementation progress (2025-11-04):* Captured DAG task skeletons for simulator smoke tests and profitability report generation, added retry + idempotency notes to the design log, and prepped fixture-backed integration tests for the dataset refresh → diagnostics export path.
   - *Release notes (2025-11-13):* `run_retraining_plan` now coordinates dataset validation, optional regime regeneration, pretraining checkpoints, supervised training, and PPO warm starts with artifact logging under a timestamped output directory, making the automation script ready for orchestration backends.
5. **Service interface** — Scaffold a FastAPI or gRPC inference service in `market_nn_plus_ultra.service` that mirrors the existing market agent contract, streams the richer Plus Ultra telemetry, and exposes curriculum configuration for reinforcement fine-tuning. **Status:** ✅ Completed — FastAPI service module plus CLI launcher now expose prediction, configuration, curriculum, and reload endpoints with telemetry-aware responses and regression coverage.
   - *Preparation (2024-02-25):* Defining optimisation KPIs (latency, throughput, calibration) that must be exposed by the service for Phase 4 monitoring.
   - *Readiness checklist (2025-10-18):* Specify service-level objectives for response latency and telemetry freshness, sketch contract tests that validate parity with the legacy market agent, and map deployment observability hooks required for Phase 4 monitoring.
   - *Interface alignment (2025-10-25):* Captured telemetry payload requirements (diagnostics aggregates, cross-asset profiling summaries, PPO curriculum metadata) to ensure the service contract lands with the metrics Phase 4 monitoring expects.
   - *Protocol evaluation (2025-10-26):* Compared gRPC vs. REST transport trade-offs, documenting compression strategies for diagnostics-heavy responses and fallback handling when calibration metadata is delayed.
   - *Kickoff planning (2025-10-29):* Scheduled design review covering streaming payload schemas, aligned authentication hooks with operations playbook requirements, and queued stub implementations for telemetry pagination ahead of full service scaffolding.
   - *Contract scoping (2025-11-01):* Drafted service payload schemas incorporating calibration-head distributions, PPO diagnostics snapshots, and simulator trade traces; confirmed pagination strategy for telemetry attachments and backward-compatible fallbacks with the legacy agent.
   - *Monitoring handshake (2025-11-02):* Synced planned service metrics (latency, calibration drift, execution slippage) with the monitoring milestone, reserving namespace allocations and alert thresholds for deployment readiness reviews.
   - *Integration sync (2025-11-03):* Documented streaming payload chunking strategy for simulator traces, scheduled contract tests covering calibration-head payload evolution, and mapped CI coverage for service/CLI parity before implementation kicks off.
   - *Implementation progress (2025-11-04):* Finalised authentication/telemetry handshake outline, captured streaming pagination retry semantics, and queued contract-test fixtures that mirror newly locked profitability reporting payloads.
   - *Landing notes (2025-11-18):* Released `market_nn_plus_ultra.service` with cached-agent management, JSON-serialisable telemetry payloads, curriculum inspection, hot-reload support, and coverage via `tests/test_service.py`, alongside `scripts/service.py` for local uvicorn launches.

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

1. **Run reporting** — Extend `scripts/generate_report.py` to incorporate backtest charts, attribution tables, scenario analysis, profitability summaries, and bootstrapped confidence intervals so every long training session outputs actionable ROI, Sharpe, and drawdown diagnostics. Emit both HTML and lightweight Markdown outputs. **Status:** ✅ Completed — reports now bundle profitability snapshots, per-symbol attribution, scenario analysis (drawdown, best/worst windows), and bootstrapped confidence intervals with refreshed documentation and regression coverage.
   - *Notes (2024-03-05):* Enumerate telemetry inputs produced by the forthcoming diagnostics callbacks, map them to report sections, and draft a sample outline so implementation can start as soon as stability tooling lands.
   - *Notes (2024-03-08):* Extend the outline to reference schema validation artefacts (per-table failure summaries, CLI strict-mode logs) so automated reports surface data-quality provenance alongside performance metrics.
   - *Implementation notes (2025-10-18):* (a) Define templating slots for calibration drift, gradient noise, and PPO warm-start diagnostics, (b) script fixture-backed snapshot tests for Markdown/HTML parity, and (c) capture performance budgets to keep report generation under five minutes for 24 GB runs.
   - *Narrative prep (2025-10-21):* Draft storyline templates that incorporate cross-asset benchmarking results and simulator calibration summaries, ensuring future reports connect feature breadth to execution realism without duplicating instrumentation work.
   - *Template sync (2025-10-26):* Linked simulator cost breakdown placeholders and calibration-head diagnostics into the draft templates, scheduling design review ahead of report automation implementation.
   - *Milestone linkage (2025-10-28):* Drafted milestone-referenced narrative stubs so automated reports can summarise roadmap progress alongside ROI metrics without manual curation.
   - *Execution kickoff (2025-10-29):* Prioritised profitability dashboard components, scoped chart-generation performance tests, and scheduled telemetry schema alignment with PPO and calibration milestones to guarantee consistent report inputs.
   - *Stability pass (2025-11-08):* Introduced a configurable charts assets directory, refreshed CLI/docs coverage, and re-ran `tests/test_reporting.py` (including the new custom-directory regression) ahead of attribution table implementation.
   - *Integration sync (2025-11-03):* Assigned owners for profitability narrative templates, logged dependency on simulator telemetry exports, and planned snapshot-test coverage for calibration-confidence tables before coding.
   - *Template detailing (2025-11-06):* Produced reference notebook rendering profitability + diagnostics composites, itemised reusable chart components (calibration fan charts, simulator cost waterfalls), and drafted regression snapshot strategy to keep Markdown/HTML parity once implementation begins.
   - *Research agenda alignment (2025-11-10):* Wired milestone annotations into Markdown/HTML reports (`MilestoneReference`), added CLI JSON ingestion, and expanded regression/docs coverage so automation can tag outputs with roadmap context.
   - *Automation update (2025-11-11):* Training loop now logs ROI/Sharpe/drawdown during validation, persists `profitability_summary.{json,md}` next to checkpoints, and exposes the metrics via `TrainingRunResult` for benchmark sweeps.
2. **Live monitoring** — Integrate Prometheus metrics and alerting hooks into the inference service. Document SLOs and alert policies in `docs/operations.md`, including live decision support for extending or branching experiments. **Status:** 🗓 Planned — deferring until the service scaffold from Phase 3 establishes telemetry contracts.
   - *Observability draft (2025-10-18):* Record required Prometheus exporters, define minimum alert set (latency, calibration drift, guardrail breaches), and align metric naming with the optimisation log for traceability.
   - *Alert cadence (2025-10-26):* Matched diagnostics sampling intervals with Prometheus scrape settings and outlined simulator-derived latency histograms for future alert rules.
   - *Dashboard spec (2025-10-28):* Listed required panels (calibration confidence, ROI trend, drawdown guardrails), alert routing rules, and dependencies on the telemetry payload contract for the inference service.
   - *Kickoff prep (2025-10-29):* Coordinated with service interface planning to reserve metric namespaces, drafted alert escalation paths tied to the operations playbook, and pencilled load-test scenarios for validating monitoring throughput once the service scaffold lands.
   - *Integration sync (2025-11-03):* Scheduled telemetry contract review with service implementers, enumerated alert-simulation scripts leveraging diagnostics fixtures, and added backlog item for Grafana dashboard seeds ahead of MVP implementation.
   - *Alert prototype planning (2025-11-06):* Drafted synthetic drift + latency replay scripts to exercise Prometheus exporters, mapped alert routing into the operations playbook outline, and defined acceptance tests covering calibration-drift pages in the forthcoming Grafana dashboards.
3. **Risk guardrails** — Build guardrail modules that enforce exposure, turnover, tail-risk thresholds, and sector/factor caps during inference, configurable via YAML, and aligned with reinforcement fine-tuning outputs. **Status:** 🗓 Planned — aligning guardrail thresholds with upcoming calibration head outputs.
   - *Design hooks (2025-10-18):* Enumerate guardrail evaluation order, map fallback actions when violations occur, and earmark simulation scenarios to regression-test guardrail logic before production rollout.
   - *Calibration alignment (2025-10-26):* Logged requirement for calibration-confidence-aware overrides and synced guardrail threshold drafts with planned reporting dashboards for consistent narratives.
   - *Implementation prep (2025-10-29):* Drafted guardrail configuration schema aligned with PPO telemetry, scheduled simulator-backed regression tests, and linked guardrail alert outputs to the live monitoring escalation plan.
   - *Integration sync (2025-11-03):* Logged dependency on service payload chunking, outlined phased rollout (offline validation → simulator-in-loop tests → live shadow mode), and earmarked documentation updates for the operations playbook once guardrails are implemented.
   - *Threshold calibration prep (2025-11-06):* Drafted Monte Carlo replay scenarios combining simulator cost envelopes with calibration-head confidence bands, identified guardrail tuning datasets for equities/crypto regimes, and mapped acceptance tests for shadow-mode deployments prior to enforcing live constraints.
4. **Analyst feedback loop** — Add review tooling in `market_nn_plus_ultra.reporting` that lets human supervisors annotate trades, flag anomalies, and record veto rationales stored in SQLite for future offline RL. **Status:** 🗓 Planned — capturing analyst workflow requirements while reinforcement hooks are designed.
   - *Workflow blueprint (2025-10-18):* Detail annotation schema extensions, draft UX flow for CLI-based annotations, and specify how feedback integrates with the reinforcement replay buffer.
   - *Review tooling prep (2025-10-26):* Identified telemetry fields (calibration confidence, simulator slippage) that analysts need for context and aligned annotation export format with upcoming reporting automation.
   - *Escalation outline (2025-10-28):* Defined override approval flow tied to monitoring alerts, captured rollback SOP referencing checkpoint lineage, and documented review cadence expectations.
   - *Kickoff actions (2025-10-29):* Prepared annotation storage schema drafts, mapped CLI prompts to analyst workflows, and queued integration tests that validate replay-buffer enrichment with human feedback metadata.
   - *Integration sync (2025-11-03):* Drafted annotation playback storyboard for profitability reports, scheduled UX review for CLI prompts, and added backlog task to align feedback exports with forthcoming continuous-retraining DAG outputs.

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

### Progress Notes — 2025-10-20

* Delivered `market_nn_plus_ultra.cli.dataset_build` with strict validation wiring, quantile overrides, and copy-safe output support so operators can regenerate regime tables without manual SQL.
* Added regression tests that exercise multi-asset fixtures through the CLI, verifying label cardinality and invalid-argument handling to guard against stale quantile caches.
* Expanded README and `docs/sqlite_schema.md` with troubleshooting guidance covering strict-mode failures, label integrity checks, and practical CLI invocation patterns.
* Itemised telemetry schema needs (gradient noise, calibration drift, throughput) so stability tooling and reporting milestones inherit a consistent metric contract.

### Progress Notes — 2025-10-22

* Implemented the cross-asset alignment engine in `market_nn_plus_ultra.data.cross_asset`, exposing rich fill-rate/dropped-row statistics for downstream profiling.
* Added `--cross-asset-view`/`--cross-asset-columns`/`--cross-asset-fill-limit` flags to the dataset-build CLI, persisting the new `cross_asset_views` table with strict validation and structured logging for instrumentation.
* Published `CROSS_ASSET_VIEW_SCHEMA` and `validate_cross_asset_view_frame` so strict mode treats the new table as a first-class citizen alongside assets, indicators, regimes, and benchmarks.
* Documented the table contract and troubleshooting guidance in `docs/sqlite_schema.md`, and introduced `scripts/benchmarks/cross_asset_profile.py` for reproducible telemetry snapshots during dataset refreshes.

### Progress Notes — 2025-10-23

* Kicked off stability tooling implementation by drafting the `TrainingDiagnosticsCallback` lifecycle, focusing on lightweight hooks for gradient-noise sampling and calibration drift measurements.
* Outlined CLI integration strategy (`--diagnostics-profile`, `--diagnostics-interval`) so researchers can enable richer telemetry without modifying experiment YAML files.
* Began mapping telemetry artefacts (parquet diagnostics, alert thresholds) into the reporting schema to maintain traceability from optimisation runs to forthcoming monitoring dashboards.

### Progress Notes — 2025-10-26

* Coordinated calibration-head planning with PPO warm-start requirements, capturing parameter sweep outlines and telemetry exports for reinforcement validation.
* Mapped execution simulator API boundaries to diagnostics cadence, ensuring reward shaping and reporting hooks share latency benchmarks.
* Synced reporting/monitoring templates with simulator cost breakdowns and calibration telemetry ahead of automation work.
* Teed up orchestration and service-interface spikes (telemetry ingestion operator, RPC boundary docs) so Sprint 4 engineering starts with stable contracts.

### Progress Notes — 2025-10-28

* Decomposed Sprint 4 backlog into experiment-ready slices covering calibration heads, PPO warm starts, profitability reporting, and monitoring dashboards.
* Registered benchmarking artefact requirements (diagnostics exports, simulator traces, cross-asset tensors) so checkpoint comparisons and architecture sweeps can run concurrently once GPU time is available.
* Expanded orchestration design notes with change-data-capture polling, calibration-drift alerting, and storage budgeting requirements in preparation for automation tasks.
* Captured service/monitoring integration hooks (streaming inference payloads, dashboard specs, escalation workflows) to reduce re-triage when implementation begins.

### Progress Notes — 2025-10-29

* Initiated Sprint 4 execution by staffing calibration-head prototyping, PPO warm-start smoke tests, profitability reporting automation, and monitoring scaffolding in parallel with shared telemetry artefacts.
* Produced day-one implementation checklists for each open milestone, covering interface sketches, fixture updates, and regression-test scaffolds required before shipping production code.
* Coordinated cross-stream dependencies by aligning simulator telemetry exports, reporting inputs, and monitoring alert schemas so downstream services can ingest new signals without schema churn.
* Scheduled mid-sprint review checkpoints to validate API surface decisions (embeddings, service payloads, guardrail configs) before deeper implementation begins.

### Progress Notes — 2025-10-30

* Implemented the calibration-aware policy head in `market_nn_plus_ultra.models.calibration`, covering quantile projections, Dirichlet concentration outputs, and parameter initialisation aligned with optimisation notes.
* Updated the Lightning training module to surface calibrated payloads through `MarketLightningModule.latest_head_output`, ensuring diagnostics and PPO warm-start planning can access confidence intervals without altering loss plumbing.
* Extended experiment configuration parsing with the `model.calibration` block (YAML + dataclass) and shipped regression tests (`tests/test_calibration_head.py`) validating quantile monotonicity, positive concentration parameters, and config round-trips.

### Progress Notes — 2025-10-25

* Consolidated cross-phase dependencies so calibration-head work, diagnostics surfacing, and PPO warm starts can proceed without double-booking engineering time.
* Drafted telemetry export schema linking diagnostics parquet outputs, cross-asset profiling metrics, and upcoming PPO rollout traces to keep reporting and monitoring milestones aligned.
* Identified documentation touchpoints (`docs/sqlite_schema.md`, forthcoming `docs/telemetry.md`) that need updates once calibration-aware heads and PPO warm starts are implemented, ensuring plan updates highlight downstream writing tasks.

### Progress Notes — 2025-10-24

* Landed the production `TrainingDiagnosticsCallback` with gradient-noise ratio tracking, calibration drift summaries, and threshold-based warnings tied to YAML/CLI toggles.
* Added regression coverage for diagnostics statistics and configuration parsing to prevent silent regressions as telemetry expands.
* Updated default experiment configs and the CLI so researchers can opt into richer diagnostics profiles or adjust alert thresholds without editing source.

### Progress Notes — 2024-02-24

* Kicked off Pandera schema enforcement work; drafting validators for core tables and planning CLI integration.
* Documented preparatory research for market-regime labelling and cross-asset feature alignment to unblock subsequent automation tasks.
* Captured pre-work for benchmarking, stability tooling, and downstream reinforcement/operations milestones so sequencing remains intact while optimisations proceed.
* Delivered the first benchmarking harness CLI (`scripts/benchmarks/architecture_sweep.py`) that records parquet catalogues for architecture sweeps, unblocking upcoming 4090 profiling sessions.

### Progress Notes — 2025-11-03

* Conducted Sprint 4 integration sync covering calibration-head exports, PPO warm-start smoke tests, simulator profiling harness expectations, and reporting/monitoring dependencies.
* Logged remaining CI/regression coverage for PPO warm starts, simulator trace exports, and profitability report templates to keep upcoming PRs tightly scoped.
* Sequenced documentation updates (`docs/sqlite_schema.md`, telemetry guide, operations playbook) alongside planned implementations so knowledge capture remains in lockstep with code delivery.
* Captured ownership for cross-stream telemetry contracts (diagnostics parquet schemas, simulator trace payloads, service response pagination) to reduce coordination overhead during implementation.

### Progress Notes — 2025-11-06

* Wrapped Sprint 4 delivery threads, recording open follow-ups for calibration export plumbing, warm-start regression triage, and documentation updates before handing off to Sprint 5 owners.
* Detailed Sprint 5 backlog slices spanning simulator package scaffolding, distributed PPO rollout profiling, profitability reporting templates, and containerisation spec drafts with explicit telemetry dependencies.
* Scheduled profiling/alerting dry runs (simulator latency replays, Prometheus exporter smoke tests) to validate instrumentation assumptions ahead of implementation tickets landing.
* Aligned operations playbook outline with forthcoming guardrail calibration and monitoring artefacts so Sprint 5 deliverables feed directly into deployment readiness narratives.

### Progress Notes — 2025-11-04

* Finalised warm-start CLI gating and ran the first PPO smoke test with calibration telemetry enabled, confirming diagnostics payloads align with the reporting schema agreed during the integration sync.
* Executed simulator kernel smoke tests on synthetic fixtures, capturing latency bucket histograms and validating telemetry export formats ahead of PPO rollout integration.
* Locked profitability reporting payload contracts (diagnostics aggregates, simulator cost deltas, calibration confidence bands) and propagated schema IDs to the service and orchestration design notes to avoid churn when automation lands.
* Stubbed orchestration DAG tasks for simulator smoke tests and profitability report generation, adding retry/idempotency notes and planning fixture-backed regression coverage for the dataset refresh → diagnostics export path.

### Progress Notes — 2025-11-10

* Landed dilation schedule enumeration in `scripts/benchmarks/architecture_sweep.py`, enabling architecture sweeps to scan horizon/depth combinations alongside MoE and state-space dilation grids from a single CLI invocation.
* Updated benchmarking fixtures and documentation hooks so parquet catalogues embed the dilation signature, unlocking downstream visualisation notebooks that compare receptive-field schedules across asset universes.
* Recorded optimisation log entry to tie the new CLI functionality back to the Phase 1 benchmarking milestone and to surface the new `--dilation-schedules` syntax for researchers scheduling 4090 profiling runs.

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

