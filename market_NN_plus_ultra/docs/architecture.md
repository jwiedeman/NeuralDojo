# Market NN Plus Ultra Architecture

This document captures the current system design for the "ultimate" trader.

## Data Interfaces

* **SQLite contract** – `assets`, `series`, `indicators`, `trades`, and
  `benchmarks` tables. The training pipeline currently consumes the first three
  while reserving room to integrate order simulation feedback in later stages.
* **Feature pipeline** – Extensible registry-driven transforms that enrich the
  price panel with technical, statistical, and spectral signals. New features
  are declared via `FeatureSpec` objects that describe dependencies, tags, and
  documentation metadata.
* **Window dataset** – Converts the multi-indexed panel into overlapping
  windows. Normalisation is performed per-feature to stabilise optimisation.

## Model Stack

1. **Patch Embedding** – Linear or convolutional projections to lift raw
   features into the model dimension while optionally reducing sequence length.
2. **Hybrid temporal layers** – Each block combines
   multi-head self-attention, dilated temporal convolutions, and a lightweight
   state-space mixer. The mixture balances global awareness and local regime
   sensitivity.
3. **Gated residual feed-forward** – Gated residual networks provide
   capacity for non-linear transformations without sacrificing stability.
4. **Forecast head** – A dense head maps the final token representation to a
   multi-step action distribution (buy / hold / sell by default). This can be
   swapped for regression, quantile, or policy outputs as needed.

## Training Loop

* **Risk-aware loss** – Combines prediction error with Sharpe and drawdown
  penalties. Plug in alternative utility functions by supplying a custom
  callable to `RiskAwareLoss`.
* **Optimisation** – AdamW with cosine annealing, gradient clipping, and
  optional mixed precision. This configuration scales to long training runs on
  modern GPUs while maintaining stability on CPUs.
* **Evaluation** – Validation losses are computed every epoch, and optional
  checkpoints are persisted for offline analysis. The evaluation module exposes
  ROI metrics suitable for leaderboards and automatic reporting.

## Extension Opportunities

* **Self-supervised pretraining** – Implement masked time-series or contrastive
  objectives that warm-start the backbone before task-specific fine-tuning.
* **Reinforcement learning** – Integrate differentiable market simulators and
  policy gradient fine-tuning to optimise directly for ROI under trading
  constraints.
* **Alternative data** – Expand the feature registry with macro, on-chain, or
  textual embeddings to capture orthogonal alpha sources.
* **Deployment** – Containerise the training/inference stack and expose service
  endpoints that mirror NeuralDojo's existing market agent API.

This blueprint should evolve alongside the task tracker. When implementing new
components, document assumptions here to keep the architectural vision aligned
with the "Plus Ultra" ambition.
