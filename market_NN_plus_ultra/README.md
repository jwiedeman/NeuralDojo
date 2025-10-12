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
    │   └── feature_pipeline.py# Feature engineering & indicator computation
    ├── models/
    │   ├── __init__.py
    │   ├── temporal_transformer.py # Deep temporal architecture definition
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
   * Copy `configs/default.yaml` and adapt for your data source, architecture depth, and training regime.
   * `train.py` will parse the configuration, build the model, and start the training loop.

4. **Iterate and evaluate**:
   * Use `market_nn_plus_ultra.evaluation.metrics` to compute risk-adjusted scores.
   * Feed results back into the task tracker to prioritise the next steps.

## Next Steps

See [`task_tracker.md`](./task_tracker.md) for the current backlog. Contributions should keep the "ultimate trader" vision in mind: exploit rich features, stay modular, and pursue ROI relentlessly.
