# Market NN Plus Ultra Architecture

This document captures the current system design for the "ultimate" trader.

## Data Interfaces

The system is designed around an opinionated but extensible SQLite contract. A
single database file can power both batch research and live inference jobs.

### Tables & Relationships

| Table        | Purpose                                                                                   |
|--------------|-------------------------------------------------------------------------------------------|
| `assets`     | Static metadata (`asset_id`, `symbol`, `sector`, `currency`). Serves as the entity spine.  |
| `series`     | Primary OHLCV candles keyed by (`timestamp`, `symbol`).                                   |
| `indicators` | Optional engineered indicators keyed by (`timestamp`, `symbol`, `name`).                  |
| `trades`     | (Planned) Executed trade logs for reinforcement learning and evaluation feedback.          |
| `benchmarks` | (Planned) Reference indices for relative performance measurement.                          |

`SQLiteMarketDataset` merges these tables into a panel with a hierarchical index
`(timestamp, symbol)` before optional enrichment via the feature pipeline. Each
extract is passed through Pandera schema validation so nulls, duplicate primary
keys, or unsorted timestamps are surfaced immediately rather than propagating
into the modelling stack.

### Feature Pipeline

The feature pipeline attaches additional signals by executing functions declared
in the `FeatureRegistry`. Each `FeatureSpec` records metadata (tags,
dependencies, descriptions) so that:

1. Experiments can select subsets of features with a single config knob.
2. Documentation can be auto-generated from the registry metadata.
3. Missing dependencies are surfaced gracefully with structured logging.

`FeaturePipeline.transform_panel` processes the dataset symbol by symbol,
returning a consistently indexed panel ready for windowing.

### Window Dataset

The `SlidingWindowDataset` slices the enriched panel into overlapping windows.
Key behaviours:

* Optional z-score normalisation fit within each window.
* Multi-step forecast horizons aligned with the trainer's objective.
* Configurable stride, enabling curriculum schedules over context length.

## Model Stack

1. **Patch Embedding** – `PatchEmbedding` lifts raw features into the model
dimension while preserving sequence length.
2. **Hybrid temporal layers** – `TemporalBlock` combines multi-head
   self-attention, dilated temporal convolutions, and residual feed-forward
   projections. The dilation schedule cycles across layers to cover long
   horizons.
3. **Forecast head** – `TemporalPolicyHead` maps the final token representation
   to a multi-step action distribution (buy / hold / sell by default). Swap this
   for regression, quantile, or reinforcement outputs as needed.

`TemporalBackboneConfig` exposes all scaling knobs (depth, heads, dilations)
through configuration files so experiments can scale up without touching code.

### Temporal Fusion Variant

The `TemporalFusionTransformer` backbone extends the stack with TFT-style
variable selection networks and gated residual blocks. Historical context flows
through a transformer encoder while a learned future context is decoded via
multi-head attention, producing horizon-aware embeddings that still align with
the standard policy head. Use `model.architecture: temporal_fusion` to select
this option and tweak `encoder_layers` / `decoder_layers` to scale depth
independently from the hybrid backbone.

### Omni-Scale Variant

For experiments demanding even more capacity, the `MarketOmniBackbone`
combines fine-grained transformer layers with:

* **Cross-scale attention** — fine tokens attend into a coarse context derived
  via learnable pooling, enabling awareness of macro regimes without losing
  microstructure detail.
* **State-space mixers** — gated depthwise convolutions mimic diagonal
  state-space models and propagate information over thousands of timesteps.
* **Dilated temporal mixers** — inherited from the hybrid backbone to
  reinforce local inductive bias.

This variant is controlled through the experiment configuration by setting
`model.architecture: omni_mixture` and tweaking the `ssm_state_dim`,
`coarse_factor`, and `cross_every` knobs.

### Mixture-of-Experts Variant

Ultra-deep studies can switch to `model.architecture: moe_transformer`, which
replaces the standard feed-forward blocks with a dense mixture-of-experts
layer. Each token is softly routed across a bank of specialised experts while
retaining the hybrid attention + convolution structure. The additional knobs
(`model.num_experts`, `model.router_dropout`, `model.ff_mult`) allow
researchers to scale capacity aggressively without incurring quadratic memory
growth from larger attention matrices. This makes it practical to explore
thousands of features and prolonged context windows on modern accelerator
hardware.

## Training Loop

* **Risk-aware loss** – `CompositeTradingLoss` combines prediction error with
  Sharpe and drawdown penalties while backpropagating through a differentiable,
  cost-aware PnL stream.
* **Optimisation** – AdamW with cosine annealing, gradient clipping, and mixed
  precision. The configuration scales to long training runs on modern GPUs while
  remaining viable on CPU for prototyping.
* **Evaluation** – Validation losses are computed every epoch, and checkpoints
  are persisted for offline analysis. The evaluation module exposes ROI metrics
  (Sharpe, Sortino, drawdown, Calmar, volatility) suitable for leaderboards and
  automatic reporting.
* **Inference agent** – `MarketNNPlusUltraAgent` mirrors the NeuralDojo market
  agent loop, restoring checkpoints, running sliding-window inference over
  SQLite data, and emitting prediction frames plus risk analytics.
* **Self-supervised warm start** – `MaskedTimeSeriesLightningModule` performs
  masked reconstruction pretraining so deep transformers and omni-scale models
  see large context windows before supervised ROI optimisation. Use
  `scripts/pretrain.py` with `configs/pretrain.yaml` to generate these
  checkpoints.

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
