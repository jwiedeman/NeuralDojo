# Market NN Plus Ultra Architecture

This document captures the current system design for the "ultimate" trader.

## Data Interfaces

The system is designed around an opinionated but extensible SQLite contract. A
single database file can power both batch research and live inference jobs.

### Tables & Relationships

| Table        | Purpose                                                                                   |
|--------------|-------------------------------------------------------------------------------------------|
| `assets`     | Static metadata (`asset_id`, `symbol`, `sector`, `currency`). Serves as the entity spine.  |
| `series`     | Primary OHLCV candles keyed by (`timestamp`, `asset_id`).                                  |
| `indicators` | Long-form engineered indicators with (`name`, `value`). Can be sparsely populated.         |
| `trades`     | (Planned) Executed trade logs for reinforcement learning and evaluation feedback.          |
| `benchmarks` | (Planned) Reference indices for relative performance measurement.                          |

The loader merges these tables into a panel with a hierarchical index
`(asset_id, timestamp)` before optional enrichment via the feature pipeline.

### Feature Pipeline

The feature pipeline attaches additional signals by executing functions declared
in the `FeatureRegistry`. Each `FeatureSpec` records metadata (tags,
dependencies, descriptions) so that:

1. Experiments can select subsets of features with a single config knob.
2. Documentation can be auto-generated from the registry metadata.
3. Missing dependencies are surfaced gracefully with structured logging.

### Window Dataset

The `SlidingWindowDataset` slices the enriched panel into overlapping windows.
Key behaviours:

* Optional z-score normalisation fit across the training corpus.
* Multi-step forecast horizons aligned with the trainer's objective.
* Configurable stride, enabling curriculum schedules over context length.

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
