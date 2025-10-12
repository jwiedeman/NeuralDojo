# Market NN Plus Ultra — Task Tracker

This tracker organises the roadmap toward a production-ready "ultimate trader". Tasks are grouped by stage and prioritised for compounding ROI.

## 0. Foundations (Immediate)
- [ ] Finalise SQLite schema contract for `assets`, `series`, `indicators`, `trades`, and `benchmarks` tables.
- [x] Implement feature registry with pluggable indicator computation (TA-Lib, spectral, macro, alternative data).
- [ ] Add data validation suite (Great Expectations or pandera) to guarantee clean inputs.
- [ ] Define baseline experiment YAML files for different asset classes (equities, crypto, forex).
- [ ] Document canonical data flow in `docs/architecture.md` (extend with schema diagrams).

### Feature Registry Enhancements
- [ ] Auto-generate documentation from `FeatureRegistry.describe()` into Markdown.
- [ ] Add alternative data connectors (on-chain metrics, macro calendars) into the registry.
- [ ] Surface dependency errors with structured logging for experiment reproducibility.
- [x] Seed registry with higher-moment statistics and spectral energy factors.

## 1. Deep Temporal Modelling
- [ ] Implement TemporalFusionTransformer-style encoder with multi-horizon support.
- [ ] Add multi-resolution attention block mixing dilated convolutions and transformer layers.
- [ ] Integrate state-space (S4/SSM) module for long-context retention.
- [ ] Pretrain on masked time-series reconstruction before supervised fine-tuning.

## 2. Trading Objective & Reinforcement Learning
- [ ] Add differentiable PnL simulator with position sizing, transaction costs, and slippage.
- [ ] Implement utility-aware losses (Sharpe, Sortino, drawdown penalties) in `models.losses`.
- [ ] Wrap policy gradient (PPO/IMPALA) fine-tuning on top of the supervised forecaster.
- [ ] Support batched scenario simulations to stress-test policies.

## 3. Evaluation & Monitoring
- [ ] Build evaluation harness for daily/weekly backtests with walk-forward splits.
- [ ] Compute ROI, max drawdown, Calmar, hit rate, and tail risk metrics by default.
- [ ] Generate Markdown/HTML reports with charts for quick inspection.
- [ ] Wire up experiment tracking (Weights & Biases or MLflow) for metadata and artefacts.
- [ ] Add automated guardrail metrics (exposure, turnover, tail percentiles) for live trading readiness.

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
