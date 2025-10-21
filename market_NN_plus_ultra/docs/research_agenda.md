# Market NN Plus Ultra — Research Agenda

This agenda sketches a staged delivery plan for the "ultimate" trader. Each phase is designed to compound capability while
keeping the stack maintainable and research-friendly.

## Phase 0 — Data & Foundations

* Finalise the SQLite contract covering assets, OHLCV series, engineered indicators, optional alternative data tables, and
  benchmark returns.
* Harden ingestion with schema validation, rich logging, and reproducibility hooks (checksums, snapshot manifests).
* Build automated documentation for the feature registry so researchers can audit available signals and their provenance.

## Phase 1 — Representation Learning

* Deliver masked reconstruction and contrastive InfoNCE pretraining (shipped via `scripts/pretrain.py`) so deep transformers start
  from context-aware weights before ROI fine-tuning.
* Experiment with hybrid backbones that combine multi-head attention, dilated temporal convolutions, and state-space mixers to
  stretch context windows into the multi-thousand timestep range.
* Deploy the omni-scale backbone variant that layers cross-resolution attention and gated state-space mixers for experiments
  targeting extreme horizon lengths and dense feature sets.
* ✅ Add curriculum schedules for window lengths and forecast horizons to encourage the model to learn both micro and macro
  structures without destabilising optimisation (`data.curriculum` + training `CurriculumCallback`).

## Phase 2 — Trading Objective Alignment

* ✅ Integrate differentiable PnL simulators with position sizing, friction (fees, slippage), and risk budgets to align training
  signals with deploy-time objectives (`market_nn_plus_ultra.trading.pnl`).
* ✅ Implement utility-aware losses (Sharpe/Sortino, drawdown penalties, downside deviation) and calibration-aware action heads
  (Dirichlet or quantile forecasts) to control risk appetite explicitly — losses ship alongside the new `CalibratedPolicyHead`,
  exposing quantile confidence intervals and Dirichlet concentration metrics via configuration toggles.
* ✅ Layer reinforcement learning fine-tuning (PPO-style) on top of the pretrained backbone, leveraging batched scenario
  generation via `_collect_rollout` to explore alternative market regimes.

## Phase 3 — Evaluation & Monitoring

* Stand up a walk-forward evaluation harness covering daily, weekly, and monthly cadences with rolling retraining windows.
* ✅ Produce auto-generated Markdown/HTML reports that chart ROI, drawdowns, turnover, and regime attribution for every run — regime-specific attribution tables now surface volatility/liquidity buckets alongside risk metrics in both Markdown and HTML exports.
* ✅ Stream experiment telemetry to Weights & Biases so every Plus Ultra experiment carries searchable configs, metrics, and
  artefacts. Offline logging is available for air-gapped studies.

## Phase 4 — Automation & Deployment

* Containerise training and inference with GPU acceleration, ready for orchestration in Kubernetes or managed batch runners.
* Mirror the existing NeuralDojo market agent API with a new Plus Ultra service that can operate in paper-trading and
  production modes.
* Implement online monitoring with drift detection, guardrails (max exposure, VaR limits), and human-in-the-loop overrides for
  risk officers.

---

Each phase should feed discoveries back into the task tracker. Treat the agenda as a living document—update it when new alpha
sources emerge or when practical constraints demand a change in priorities.
