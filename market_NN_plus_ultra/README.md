# Market NN Plus Ultra

Market NN Plus Ultra is a research playground dedicated to building the most capable trading agent possible inside NeuralDojo. The project focuses on:

* **Massive signal intake** from SQLite price databases containing OHLCV series, derived indicators, and alternative data.
* **State-of-the-art sequence modelling** with deep transformer-style architectures, temporal convolution modules, and self-supervised objectives.
* **Continuous evaluation and deployment loops** that mirror the existing market agent but push toward higher ROI through advanced reinforcement fine-tuning.

This folder provides an initial scaffold, documentation, and technical roadmap for delivering an "ultimate" trader that can be supplied with a SQLite database and autonomously analyse, decide, and iterate on strategies.

## Vision

1. **Rich Market Memory** — ingest long price histories, technical indicators, and curated features with minimal friction.
2. **Powerful Temporal Modeller** — leverage modern sequence models (transformers with multi-resolution attention and state-space layers) to capture non-linear dynamics.
3. **Adaptive Trading Brain** — mix supervised price/return prediction with reinforcement learning from simulated trades to directly optimise ROI.
4. **Automated Evaluation Loop** — continuously backtest against fresh data, compare baselines, and produce actionable diagnostics.

## Repository Layout

```
market_NN_plus_ultra/
├── README.md                  # Vision and orientation
├── task_tracker.md            # Living backlog for future work
├── pyproject.toml             # Python package definition + dependencies
├── configs/
│   └── default.yaml           # Example configuration for experiments
├── scripts/
│   └── train.py               # Entry point for running training pipelines
└── market_nn_plus_ultra/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── sqlite_loader.py   # Utilities for extracting structured tensors from SQLite
    │   ├── feature_pipeline.py# Feature engineering & indicator computation
    │   └── window_dataset.py  # Sliding-window dataset with z-score normalisation
    ├── models/
    │   ├── __init__.py
    │   ├── temporal_transformer.py # Hybrid transformer/state-space backbone
    │   └── losses.py          # Custom loss functions for risk-aware optimisation
    ├── training/
    │   ├── __init__.py
    │   ├── config.py          # Typed experiment configuration objects
    │   └── train_loop.py      # Orchestrates the end-to-end training lifecycle
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py         # ROI, drawdown, Sharpe, and policy evaluation metrics
    └── utils/
        ├── __init__.py
        └── seeding.py         # Reproducibility helpers
```

Each module comes with docstrings explaining expected behaviour so future contributors can quickly implement and extend components. The goal is to offer a batteries-included base that can scale from rapid experimentation on a laptop to large-scale multi-GPU studies without major refactors.

## Core Capabilities

* **Deep temporal modelling** — The default backbone stacks sixteen hybrid transformer layers that mix global attention, dilated convolutions, and state-space inspired mixers for memory retention over thousands of timesteps. Patch embeddings and learned positional encodings are designed to support high-dimensional feature spaces out of the box.
* **Rich feature engineering** — The feature registry encapsulates momentum, volatility, regime, and spectral features while remaining easily extensible. Adding new indicators only requires registering a `FeatureSpec` with dependency metadata inside `feature_pipeline.py`.
* **Risk-aware optimisation** — Custom loss functions marry standard regression objectives with differentiable Sharpe and drawdown penalties, rewarding policies that maximise return while respecting risk budgets.
* **Automated evaluation** — The evaluation module exposes risk-adjusted metrics (Sharpe, Sortino, Calmar, drawdown) and trade-level analytics that plug directly into backtesting or live monitoring loops.
* **Research ergonomics** — YAML-driven experiment configs, dataclass-backed runtime configs, and a high-level trainer streamline iteration while keeping experiments reproducible.

## Data Flow Overview

1. **SQLite ingestion** — `SQLiteMarketDataset` loads OHLCV candles, merges optional indicator tables, and restricts the universe to configured symbols.
2. **Feature augmentation** — `FeaturePipeline` executes the registered feature functions, automatically skipping features when dependencies are missing and appending results to the panel.
3. **Windowing & normalisation** — `SlidingWindowDataset` converts the multi-indexed panel into sliding windows with optional z-score normalisation, surfacing tensors ready for GPU training.
4. **Model forward pass** — `TemporalBackbone` consumes windows, applies patch embeddings, and processes sequences through the hybrid temporal stack to forecast multi-step actions/returns.
5. **Loss & optimisation** — The trainer computes risk-aware losses, backpropagates gradients, clips norms, and applies cosine-annealed AdamW updates.
6. **Evaluation loop** — Validation splits are evaluated every epoch, logging ROI-focused metrics and saving checkpoints for further analysis.

## Architecture Blueprint

The default configuration is intentionally overprovisioned to unlock a rich hypothesis space:

* **Input dimension** — Supports 128+ raw and engineered features per timestep, making room for alternative data and synthetic signals.
* **Depth & heads** — Sixteen layers with eight attention heads per layer ensure strong expressivity across temporal resolutions.
* **Multi-resolution mixing** — Convolutional dilations (`1, 2, 4, 8, 16, 32, ...`) are cycled across blocks to model both intraday microstructure and multi-week regimes.
* **Forecast horizon** — Default head predicts a five-step output tensor suitable for regression; toggle `model.output_dim` to model discrete buy/hold/sell actions or richer targets.
* **Extension points** — Swap in Temporal Fusion Transformer modules, S4-style state-space layers, or reinforcement learning policy heads without changing the trainer API.

For a deeper discussion of the data flow, modular boundaries, and suggested extensions see [`docs/architecture.md`](./docs/architecture.md).

## Getting Started

1. **Create a virtual environment** and install the package in editable mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Prepare market data**:
   * Provide a SQLite database that includes `assets`, `series`, and `indicators` tables (see `sqlite_loader.py` for the expected schema).
   * Extend `feature_pipeline.py` to compute additional technical features (e.g. Fourier transforms, macro factors, news sentiment embeddings).

3. **Configure an experiment**:
   * Copy `configs/default.yaml` and adapt for your data source, architecture depth, window size, and trading horizon.
    * Key knobs:
      - `model.conv_dilations` and `model.depth` scale temporal coverage.
      - `data.window_size` / `data.stride` control how much history flows into each batch.
      - `optimizer.lr` / `weight_decay` tune the AdamW optimiser.
    * `scripts/train.py` parses the YAML, builds dataclass configs, and launches the Lightning-powered hybrid transformer trainer.

4. **Iterate and evaluate**:
   * Use `market_nn_plus_ultra.evaluation.metrics` to compute risk-adjusted scores on trade logs.
   * Inspect checkpoints saved under `training.checkpoint_dir` and feed results back into the task tracker to prioritise the next steps.

## Model Highlights

* **Hybrid temporal backbone** — stacks multi-head attention, dilated temporal convolutions, and state-space mixers for long-term memory.
* **Rich data preprocessing** — SQLite ingestion joins OHLCV, indicators, and on-the-fly engineered features with z-score normalisation.
* **Risk-aware objectives** — differentiable Sharpe/drawdown penalties baked into the default loss encourage ROI-focussed behaviour.

## Next Steps

See [`task_tracker.md`](./task_tracker.md) for the current backlog and [`docs/research_agenda.md`](./docs/research_agenda.md) for a narrative roadmap that ties milestones together. Contributions should keep the "ultimate trader" vision in mind: exploit rich features, stay modular, and pursue ROI relentlessly.

## Feature Registry

Market NN Plus Ultra ships with a pluggable [`FeatureRegistry`](./market_nn_plus_ultra/data/feature_pipeline.py) that catalogues rich technical signals and engineered factors. Pipelines can cherry-pick subsets of indicators or extend the registry with custom research ideas. The default registry includes:

* Momentum: RSI, MACD histogram and signal differentials, multi-step price velocity.
* Volatility & risk: Bollinger band width, annualised realised volatility, Average True Range, rolling skew/kurtosis.
* Regime detection: Soft bull/bear probability estimates from rolling z-scores.
* Volume analytics: Rolling z-score anomalies for volume-led confirmation.
* Distribution & spectral cues: Log returns and FFT energy decomposition to expose multi-scale structure.

The registry metadata powers automated documentation and enables future UIs to surface feature provenance, required dependencies, and semantic tags.
