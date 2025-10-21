# Market NN Plus Ultra

Market NN Plus Ultra is a research playground dedicated to building the most capable trading agent possible inside NeuralDojo. The project focuses on:

* **Massive signal intake** from SQLite price databases containing OHLCV series, derived indicators, and alternative data.
* **State-of-the-art sequence modelling** with deep transformer-style architectures, temporal convolution modules, and self-supervised objectives.
* **Continuous evaluation and deployment loops** that mirror the existing market agent but push toward higher ROI through advanced reinforcement fine-tuning.

This folder provides an initial scaffold, documentation, and technical roadmap for delivering an "ultimate" trader that can be supplied with a SQLite database and autonomously analyse, decide, and iterate on strategies.

## Quickstart

The commands below take you from a fresh clone to running both training and inference. They assume you are inside the `market_NN_plus_ultra/` directory.

1. **Create a virtual environment and install the package**

   <details>
   <summary><strong>macOS / Linux (bash, zsh)</strong></summary>

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .
   ```

   </details>

   <details>
   <summary><strong>Windows (PowerShell)</strong></summary>

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -e .
   ```

   ```powershell
   # Optional: install the CUDA-enabled PyTorch wheel
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

   </details>

   Installing in editable mode registers the `market_nn_plus_ultra` package and pulls in required dependencies such as PyTorch Lightning, pandas, and SQLAlchemy. This resolves the `ModuleNotFoundError` errors you would see when running the scripts without installing the project first. Confirming `torch.cuda.is_available()` returns `True` ensures Lightning keeps the GPU accelerator without falling back to CPU.

2. **Provide market data**

   The quickest path is to run the bundled bootstrap script which downloads OHLCV candles for a basket of liquid ETFs and seeds the SQLite database::

   ```bash
   python scripts/bootstrap_sqlite.py --db-path data/market.db \
       --tickers SPY QQQ VTI IWM EFA EEM XLK XLF XLY XLP \
       --start 2000-01-01 --end 2025-01-01
   ```

   The script is idempotent; re-run it to refresh data or pass `--overwrite` to rebuild the tables. Alternatively copy a SQLite database that matches the schema documented in [`docs/sqlite_schema.md`](./docs/sqlite_schema.md) into `data/market.db`, or update the `data.sqlite_path` entry in your config YAML to point at your file. At minimum the database should contain `assets`, `series`, and any indicator tables you reference in the config.

   Need a high-variance synthetic dataset for GPU benchmarking? Generate one with
   the new fixture builder which fuses price history, technicals, alternative
   signals, and regime labels in a single command:

   ```bash
   python scripts/make_fixture.py data/plus_ultra_fixture.db \
       --symbols SPY QQQ IWM BTC-USD ETH-USD \
       --rows 32768 --freq 15min --alt-features 4
   ```

   The resulting SQLite file drops straight into `data.sqlite_path` and is fully
   validated by Pandera before it is written to disk.

   Regenerate regime labels or run integrity checks against an existing SQLite
   bundle at any time with the dataset-build CLI:

   ```bash
   python -m market_nn_plus_ultra.cli.dataset_build data/market.db \
       --regime-labels --strict-validation
   ```

   To sanity check connectivity, drop into a Python shell and load the first few rows:

   ```bash
   # macOS / Linux
   python - <<'PY'
   from market_nn_plus_ultra.data.sqlite_loader import SQLiteMarketDataset, SQLiteMarketSource

   dataset = SQLiteMarketDataset(SQLiteMarketSource(path="data/market.db"), validate=False)
   frame = dataset.load()
   print(frame.head())
   PY
   ```

   ```powershell
   # Windows PowerShell
   python -c "from market_nn_plus_ultra.data.sqlite_loader import SQLiteMarketDataset, SQLiteMarketSource; "^
   "dataset = SQLiteMarketDataset(SQLiteMarketSource(path='data/market.db'), validate=False); "^
   "frame = dataset.load(); print(frame.head())"
   ```

3. **(Optional) Warm start with self-supervised pretraining**

   ```bash
   python scripts/pretrain.py --config configs/pretrain.yaml --accelerator cpu --devices 1 --max-epochs 1
   ```

   ```powershell
   python scripts/pretrain.py --config configs/pretrain.yaml --accelerator gpu --devices 1 --max-epochs 1
   ```

   Adjust the overrides (e.g. `--accelerator gpu`, `--devices 1`) to match your hardware. The command saves checkpoints under `checkpoints/pretrain/` by default and now auto-launches a Weights & Biases run with sensible defaults so you can watch loss curves evolve in real time. Add `--wandb-offline` for air-gapped environments or `--no-wandb` if you need to suppress tracking entirely.

4. **Train the supervised model**

   ```bash
   python scripts/train.py --config configs/default.yaml --accelerator cpu --devices 1 --max-epochs 1
   ```

   ```powershell
   python scripts/train.py --config configs/default.yaml --accelerator gpu --devices 1 --max-epochs 1
   ```

   Remove or change the CLI overrides once you are ready to run a full GPU-backed training session. Checkpoints land in `checkpoints/default/`. As with pretraining, Weights & Biases logging starts automatically (project name defaults to `plus-ultra` and run names include the config + timestamp). Opt out with `--no-wandb` if you prefer local-only console logging.

   > 💡 Running into GPU memory pressure on a consumer card? Swap in `configs/default_desktop.yaml` for a lighter-weight architecture (`--config configs/default_desktop.yaml`). The desktop preset trims the model depth, attention width, and batch size, and disables persistent dataloader workers so training stays responsive on Windows.

   Warm start the supervised run from a pretraining checkpoint at any time with:

   ```bash
   python scripts/train.py --config configs/default.yaml \
       --pretrain-checkpoint checkpoints/pretrain/best.ckpt
   ```

5. **Run the inference agent**

   ```bash
   python scripts/run_agent.py --config configs/default.yaml \
     --checkpoint checkpoints/default/last.ckpt \
     --device cpu \
     --output outputs/predictions.parquet
   ```

   ```powershell
   python scripts/run_agent.py --config configs/default.yaml `
     --checkpoint checkpoints/default/last.ckpt `
     --device cuda:0 `
     --output outputs\predictions.parquet
   ```

   Swap in the path to the checkpoint you want to evaluate (`last.ckpt`, `best.ckpt`, etc.) and, if desired, request GPU execution with `--device cuda:0`. The script writes predictions to the requested parquet (or CSV) file and prints evaluation metrics when realised returns are available.

6. **Benchmark architecture variants**

   ```bash
   python scripts/benchmarks/architecture_sweep.py \
       --config configs/default.yaml \
       --architectures hybrid_transformer,omni \
       --model-dims 512,768 \
       --depths 8,12 \
       --limit-train-batches 0.1 \
       --max-epochs 1
   ```

   The benchmarking CLI disables Weights & Biases logging by default, runs each scenario sequentially, and emits a parquet catalogue under `benchmarks/` capturing dataset stats, runtime, and validation metrics for downstream analysis. Use `--enable-wandb` when you want each run to report into your tracking workspace.

7. **Compare pretraining warm starts against scratch runs**

   ```bash
   python scripts/benchmarks/pretraining_comparison.py \
       --config configs/default.yaml \
       --pretrain-checkpoint checkpoints/pretrain/best.ckpt \
       --output benchmarks/pretraining_comparison.parquet
   ```

   The CLI runs two supervised jobs (one from scratch, one seeded by the pretraining checkpoint), aggregates metrics into a parquet catalogue, and writes checkpoints to `benchmarks/pretraining_runs/`. A third row records warm-start minus scratch deltas for every numeric metric, and the command prints a concise summary of the most useful deltas (validation loss, ROI, Sharpe, runtime) so you can gauge warm-start lift at a glance. Use the output to quantify validation lift, ROI changes, or training-time savings before rolling the warm start into production recipes.

8. **Evaluate stability with walk-forward splits**

   Once you have predictions with realised returns, run the walk-forward
   evaluation CLI to generate per-split metrics and an aggregated summary:

   ```bash
   python scripts/walkforward_eval.py \
       --predictions outputs/predictions.parquet \
       --train-window 252 \
       --test-window 63 \
       --metrics-output benchmarks/walkforward_metrics.parquet \
       --summary-output benchmarks/walkforward_summary.json
   ```

   ```powershell
   python scripts/walkforward_eval.py `
       --predictions outputs\predictions.parquet `
       --train-window 252 `
       --test-window 63 `
       --metrics-output benchmarks\walkforward_metrics.parquet `
       --summary-output benchmarks\walkforward_summary.json
   ```

   The command emits a per-split metrics table and records aggregate Sharpe,
   drawdown, and hit-rate statistics, helping you track deployment readiness as
   you iterate on training runs.

9. **Automate retraining with the orchestration CLI**

   ```bash
   python scripts/automation/retrain.py \
       --dataset data/market.db \
       --train-config configs/default.yaml \
       --pretrain-config configs/pretrain.yaml \
       --run-reinforcement \
       --warm-start training
   ```

   The command above validates the SQLite dataset, optionally regenerates regime labels, runs pretraining, launches supervised training, and finishes with PPO fine-tuning. Artifacts land under `automation_runs/<timestamp>/`, including checkpoints, profitability summaries, and policy state dicts. Add flags such as `--regenerate-regimes`, `--skip-pretraining`, or `--skip-training` to tailor individual stages.

   Looking for a complete walkthrough that chains fixture creation, pretraining,
   training, evaluation, and monitoring? Follow the
   [end-to-end MVP quickstart](./docs/quickstart_mvp.md) to run the full loop
   and capture Prometheus-ready telemetry after each pass.

10. **Serve the inference API**

   ```bash
   python scripts/service.py --config configs/default.yaml --host 0.0.0.0 --port 8000
   ```

   The FastAPI service loads the same experiment configuration as the CLI agent and exposes `/health`, `/config`, `/curriculum`, `/predict`, and `/reload` endpoints. Override defaults with flags such as `--checkpoint`, `--device`, or `--max-prediction-rows` when deploying. Responses include telemetry snapshots (feature columns, horizon, window size) so monitoring dashboards can ingest predictions without extra glue code.

## Containerised workflow

Prefer a reproducible runtime with CUDA already configured? Build the GPU-ready
image under [`docker/Dockerfile.gpu`](./docker/Dockerfile.gpu) and run training,
pretraining, the offline agent, or the FastAPI service with a single command.
The entrypoint accepts the same sub-commands as the native scripts:

```bash
docker buildx build -f docker/Dockerfile.gpu -t plus-ultra:latest .

# Launch supervised training on the GPU
docker run --gpus all --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  plus-ultra:latest \
  train --config configs/default.yaml --accelerator gpu --devices 1

# Serve the FastAPI inference API
docker run --gpus all --rm -p 8000:8000 \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  plus-ultra:latest \
  service --config configs/default.yaml --checkpoint checkpoints/default/last.ckpt
```

See [`docs/containerization.md`](./docs/containerization.md) for detailed
volume recommendations, troubleshooting tips, and additional entrypoint
examples.

## Windows vs. WSL

You can run the project either directly in Windows or inside Windows Subsystem for Linux (WSL):

* **Native Windows (PowerShell)** — Recommended when you want to leverage NVIDIA's CUDA drivers with the official PyTorch wheels. Install Python 3.11, activate the virtual environment, and add the `cu126` wheel as shown above. PowerShell uses backticks (`` ` ``) for line continuations, so the README provides explicit Windows variants for multi-line commands.
* **WSL (Ubuntu/Debian)** — Offers a Linux userland with GPU passthrough from Windows. Follow the macOS/Linux instructions inside the WSL terminal and install the CUDA 12.6-compatible wheel (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`). WSL supports bash heredocs like `python - <<'PY'`.

Pick whichever workflow fits your tooling preferences; both are compatible with Lightning once the environment is initialised and the package is installed in editable mode.

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
│   ├── default.yaml           # Supervised training example configuration
│   ├── pretrain.yaml          # High-capacity self-supervised pretraining recipe
│   └── pretrain_desktop.yaml  # Consumer-GPU friendly pretraining preset
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
    │   ├── moe_transformer.py      # Mixture-of-experts expansion of the hybrid stack
    │   ├── omni_mixture.py      # Omni-scale backbone with cross-resolution attention
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
* **Temporal fusion option** — A Temporal Fusion Transformer style backbone mixes variable selection networks, gated residual blocks, and encoder-decoder attention to deliver high-capacity multi-horizon forecasting with a strong inductive bias for tabular time-series.
* **Omni-scale backbone option** — The `omni_mixture` architecture fuses fine-grained transformer layers with coarse cross-scale attention, gated state-space mixers, and dilated convolutions to capture both microstructure and macro regime shifts in one model.
* **State-space backbone option** — The `state_space` architecture distils modern structured state-space mixers (S4/S5-style) into a lightweight residual stack that excels at thousand-timestep contexts while remaining GPU friendly.
* **Mixture-of-experts scalability** — The `moe_transformer` variant routes tokens through a bank of specialised feed-forward experts, unlocking extra capacity without ballooning quadratic attention cost.
* **Rich feature engineering** — The feature registry encapsulates momentum, volatility, regime, and spectral features while remaining easily extensible. Adding new indicators only requires registering a `FeatureSpec` with dependency metadata inside `feature_pipeline.py`.
* **Risk-aware optimisation** — Custom loss functions marry standard regression objectives with differentiable Sharpe and drawdown penalties, rewarding policies that maximise return while respecting risk budgets.
* **Differentiable PnL simulator** — Training losses can now backpropagate through a cost-aware PnL stream that includes transaction, slippage, and holding penalties for realistic signal shaping.
* **Reinforcement fine-tuning** — PPO-style policy optimisation on top of supervised checkpoints lets the agent learn directly from differentiable PnL trajectories with configurable trading costs.
* **Automated evaluation** — The evaluation module exposes risk-adjusted metrics (Sharpe, Sortino, Calmar, drawdown), regime-aware attribution tables, and trade-level analytics that plug directly into backtesting or live monitoring loops.
* **Walk-forward harness** — `WalkForwardBacktester` enables regimented daily/weekly/monthly evaluation splits, automatically summarising ROI-focused metrics per regime.
* **Research ergonomics** — YAML-driven experiment configs, dataclass-backed runtime configs, and a high-level trainer streamline iteration while keeping experiments reproducible.

## Data Flow Overview

1. **SQLite ingestion** — `SQLiteMarketDataset` loads OHLCV candles, merges optional indicator tables, and restricts the universe to configured symbols while running Pandera-backed schema checks to catch nulls, duplicates, or unsorted timestamps before they pollute training runs.
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
The high-level multi-phase execution plan lives in [`docs/implementation_plan.md`](./docs/implementation_plan.md) and maps the
task tracker backlog into chronological milestones.

## Getting Started

1. **Create a virtual environment and install the package**

   Choose the commands that match your shell. The Windows instructions assume PowerShell and Python 3.11 (the newest release with official CUDA wheels as of 2025).

   <details>
   <summary><strong>macOS / Linux (bash, zsh)</strong></summary>

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .
   ```

   </details>

   <details>
   <summary><strong>Windows (PowerShell)</strong></summary>

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -e .
   ```

   </details>

   Installing in editable mode registers the `market_nn_plus_ultra` package and pulls in required dependencies such as PyTorch Lightning, pandas, and SQLAlchemy. This resolves the `ModuleNotFoundError` errors you would see when running the scripts without installing the project first.

   *Windows GPU wheel:* if you want CUDA acceleration on bare-metal Windows, install the CUDA-enabled torch wheel **after** activating the virtual environment:

   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

   The `cu126` index currently publishes wheels for Python 3.11. If `torch.cuda.is_available()` prints `False`, double-check that the virtual environment is active and that you are not on an unsupported Python version (e.g. 3.13). When the command returns `True`, Lightning will keep the `gpu` accelerator without falling back to CPU.

2. **Prepare market data**:
   * Provide a SQLite database that includes `assets`, `series`, and `indicators` tables (see `sqlite_loader.py` for the expected schema).
   * Extend `feature_pipeline.py` to compute additional technical features (e.g. Fourier transforms, macro factors, news sentiment embeddings).
   * Reference [`docs/sqlite_schema.md`](./docs/sqlite_schema.md) for the canonical table definitions and indexing guidance.

3. **Configure an experiment**:
   * Copy `configs/default.yaml` and adapt for your data source, architecture depth, window size, and trading horizon.
    * Key knobs:
      - `model.conv_dilations` and `model.depth` scale temporal coverage.
      - `data.window_size` / `data.stride` control how much history flows into each batch.
      - `optimizer.lr` / `weight_decay` tune the AdamW optimiser.
    * `scripts/train.py` parses the YAML, builds dataclass configs, and launches the Lightning-powered hybrid transformer trainer.

4. **Warm-start with self-supervision (optional but recommended)**:
   * Launch the self-supervised pretraining loop with `python scripts/pretrain.py --config configs/pretrain.yaml` to initialise deep weights before supervised optimisation.
   * Choose between masked reconstruction (`--objective masked`) and contrastive TS2Vec-style pretraining (`--objective contrastive`). The masked variant replaces random timesteps and learns to in-fill them, while the contrastive route generates augmented views (jitter, scaling, time masking) and maximises agreement through an InfoNCE loss.
   * Pass overrides such as `--devices auto`, `--max-epochs 100`, `--mask-prob 0.35`, or `--temperature 0.05` directly to the CLI to iterate quickly without editing YAML files. When `--mask-value mean` or `--time-mask-fill mean` is supplied the loader fills masked regions with the window average.

5. **Iterate and evaluate**:
   * Use `market_nn_plus_ultra.evaluation.metrics` to compute risk-adjusted scores on trade logs.
   * Inspect checkpoints saved under `training.checkpoint_dir` and feed results back into the task tracker to prioritise the next steps.
   * Deploy the new inference agent (`scripts/run_agent.py`) to mirror the classic market agent workflow on fresh SQLite dumps. The CLI now streams a formatted risk dashboard to stdout and can persist metrics to JSON/CSV/Markdown via `--metrics-output`.
   * Profile backbone throughput and memory with `scripts/profile_backbone.py --architecture omni_mixture --seq-len 1024 --batch-size 32` to validate hyper-parameters against available hardware before launching long experiments.
   * Run `python scripts/ci_compile.py --quiet` as a lightning-fast syntax smoke test before longer training or linting cycles. The command wraps `compileall` so it is trivial to drop into CI.
* Produce investor-ready performance summaries with `scripts/generate_report.py --predictions outputs/predictions.parquet --output reports/latest.html`. The command renders Markdown or HTML, complete with risk metrics and charts, for rapid experiment reviews. When your predictions include a benchmark column, add `--benchmark-column benchmark_return` to surface excess return, tracking error, information ratio, and beta alongside the default ROI metrics.
   * Stream telemetry to Weights & Biases by setting `wandb_project` in your YAML or passing `--wandb-project plus-ultra` on the CLI. Additional overrides (`--wandb-tags`, `--wandb-offline`, etc.) make it easy to log runs even when working offline.
   * Regenerate feature documentation with `python scripts/export_features.py --output docs/generated_features.md` after extending the registry.

### Running the Inference Agent

Once a model has been trained (or even with random weights for smoke tests) the Plus Ultra agent can be executed directly against a SQLite database:

```bash
python scripts/run_agent.py --config configs/default.yaml --checkpoint path/to/checkpoint.ckpt --device cuda:0 \
  --output outputs/predictions.parquet
```

Training runs accept the same style of overrides. For example, to train an omni backbone with 24 layers and a widened model dimension on a single GPU without touching the YAML:

```bash
 python scripts/train.py --config configs/default.yaml --devices 1 --architecture omni_mixture \
  --depth 24 --model-dim 1024 --batch-size 64 --learning-rate 0.0001
```

The script orchestrates data loading, feature enrichment, sliding-window inference, and optional ROI evaluation if realised returns are present. Use `--no-eval` to skip metrics and `--return-column` to target a specific realised return column in the generated prediction frame.

**Diagnostics tooling:** the training CLI exposes first-class controls for the stability telemetry shipped in `configs/default*.yaml`.

* `--enable-diagnostics` / `--disable-diagnostics` toggle the callback regardless of YAML defaults.
* `--diagnostics-interval` changes how often gradient-noise statistics are sampled.
* `--diagnostics-profile` turns on extended validation summaries (calibration spread, bias variance).
* `--diagnostics-noise-threshold`, `--diagnostics-bias-threshold`, and `--diagnostics-error-threshold` adjust warning guardrails without editing config files.
* The `diagnostics` section inside the YAML mirrors these switches so long runs can opt into richer telemetry by default.

### Reinforcement Fine-Tuning

When it's time to align the agent with ROI directly, kick off policy-gradient optimisation with PPO:

```bash
python scripts/rl_finetune.py --config configs/default.yaml --checkpoint path/to/supervised.ckpt \
  --device cuda:0 --updates 25 --steps-per-rollout 512
```

The CLI mirrors the reinforcement config dataclass so you can override rollout size, discount factors, clipping ratios, and whether dataset targets are returns. Each update prints reward, policy loss, value loss, and entropy so you can track convergence at a glance.

## Model Highlights

* **Hybrid temporal backbone** — stacks multi-head attention, dilated temporal convolutions, and state-space mixers for long-term memory. A heavier `MarketOmniBackbone` pushes context even further with cross-resolution attention and gated state-space mixers.
* **Rotary-aware attention** — the hybrid backbone now ships with configurable rotary positional embeddings and an extendable `max_seq_len`, keeping extremely long price histories numerically stable while avoiding quadratic memory blow-ups.
* **Rich data preprocessing** — SQLite ingestion joins OHLCV, indicators, and on-the-fly engineered features with z-score normalisation.
* **Risk-aware objectives** — differentiable Sharpe/drawdown penalties baked into the default loss encourage ROI-focussed behaviour.
* **Self-supervised warm start** — Masked reconstruction and contrastive InfoNCE pretraining (`scripts/pretrain.py`) prime the backbone on vast context windows before ROI-optimised fine-tuning.

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
