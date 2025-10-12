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
        └── __init__.py
```

Each module comes with docstrings explaining expected behaviour so future contributors can quickly implement and extend components.

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
     - `training.window_size` / `window_stride` control how much history flows into each batch.
     - `optimizer.lr` / `weight_decay` tune the AdamW optimiser.
   * `train.py` parses the YAML, builds dataclass configs, seeds RNGs, and launches the hybrid transformer trainer.

4. **Iterate and evaluate**:
   * Use `market_nn_plus_ultra.evaluation.metrics` to compute risk-adjusted scores on trade logs.
   * Inspect checkpoints saved under `training.checkpoint_dir` and feed results back into the task tracker to prioritise the next steps.

## Model Highlights

* **Hybrid temporal backbone** — stacks multi-head attention, dilated temporal convolutions, and state-space mixers for long-term memory.
* **Rich data preprocessing** — SQLite ingestion joins OHLCV, indicators, and on-the-fly engineered features with z-score normalisation.
* **Risk-aware objectives** — differentiable Sharpe/drawdown penalties baked into the default loss encourage ROI-focussed behaviour.

## Next Steps

See [`task_tracker.md`](./task_tracker.md) for the current backlog. Contributions should keep the "ultimate trader" vision in mind: exploit rich features, stay modular, and pursue ROI relentlessly.

## Feature Registry

Market NN Plus Ultra ships with a pluggable [`FeatureRegistry`](./market_nn_plus_ultra/data/feature_registry.py) that catalogues rich technical signals and engineered factors. Pipelines can cherry-pick subsets of indicators or extend the registry with custom research ideas. The default registry includes:

* Momentum: RSI, MACD histogram and signal differentials.
* Volatility & risk: Bollinger band width, annualised realised volatility, Average True Range.
* Regime detection: Soft bull/bear probability estimates from rolling z-scores.
* Volume analytics: Rolling z-score anomalies for volume-led confirmation.

The registry metadata powers automated documentation and enables future UIs to surface feature provenance, required dependencies, and semantic tags.
