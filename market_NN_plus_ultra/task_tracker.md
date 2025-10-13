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
- [ ] Profile omni-scale backbone throughput and memory on GPU/CPU to tune default hyperparameters.
- [ ] Add CI job that exercises `python -m compileall` + unit stubs to protect against syntax drift.

### Feature Registry Enhancements
- [x] Auto-generate documentation from `FeatureRegistry.describe()` into Markdown.
- [ ] Add alternative data connectors (on-chain metrics, macro calendars) into the registry.
- [x] Surface dependency errors with structured logging for experiment reproducibility. (PR TBD)
- [x] Seed registry with higher-moment statistics and spectral energy factors.

## 1. Deep Temporal Modelling
- [x] Implement TemporalFusionTransformer-style encoder with multi-horizon support.
- [x] Add multi-resolution attention block mixing dilated convolutions and transformer layers.
- [x] Integrate state-space (S4/SSM) module for long-context retention.
- [x] Ship mixture-of-experts expansion for scaling feed-forward capacity without quadratic attention cost.
- [ ] Benchmark omni-scale backbone versus hybrid baseline across multiple asset universes.
- [x] Pretrain on masked time-series reconstruction before supervised fine-tuning (`scripts/pretrain.py`).
- [ ] Benchmark pretraining checkpoints vs. scratch initialisation across asset classes.
- [ ] Extend pretraining tasks with contrastive (TS2Vec-style) objectives for regime discrimination.
- [ ] Introduce curriculum over window sizes and horizons to stabilise very deep models.

## 2. Trading Objective & Reinforcement Learning
- [x] Add differentiable PnL simulator with position sizing, transaction costs, and slippage.
- [x] Implement utility-aware losses (Sharpe, Sortino, drawdown penalties) in `models.losses`.
- [x] Wrap policy gradient (PPO/IMPALA) fine-tuning on top of the supervised forecaster.
- [x] Support batched scenario simulations to stress-test policies.
- [ ] Explore calibration-aware heads (Dirichlet / quantile) for action confidence.

## 3. Evaluation & Monitoring
- [x] Build evaluation harness for daily/weekly backtests with walk-forward splits.
- [x] Compute ROI, max drawdown, Calmar, hit rate, and tail risk metrics by default.
- [x] Ship inference agent + CLI to produce prediction frames from SQLite sources.
- [ ] Generate Markdown/HTML reports with charts for quick inspection.
- [ ] Wire up experiment tracking (Weights & Biases or MLflow) for metadata and artefacts.
- [ ] Add automated guardrail metrics (exposure, turnover, tail percentiles) for live trading readiness.
- [ ] Publish automated run reports referencing the research agenda milestones.

## 4. Automation & Deployment
- [ ] Containerise training + inference pipelines with GPU support.
- [ ] Expose REST/gRPC service that mirrors the existing market agent API but uses the new brain.
- [ ] Schedule continuous retraining jobs triggered by new data arrival.
- [ ] Set up online monitoring for live performance and drift detection.
- [ ] Build playbook for human-in-the-loop overrides and risk manager approvals.

---

### Contributing Guidelines
* Keep tasks small and outcome-driven.
* Update statuses during each iteration.
* Link to PRs / experiments when tasks are completed.
