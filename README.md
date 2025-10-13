# NeuralGo Experiments

This repository now hosts a small suite of browser-first neural playgrounds:

- **`neuralgo/`** — the original Go self-play lab, preserved exactly as before but now layered on top of the shared UI system.
- **`NeuralMatch/`** — a new synthetic match-3 evaluator that trains on aggressively augmented puzzle layouts.
- **`NeuralStocks/`** — a market forecaster playground that streams SPY closes through a compact MLP and visualises the learning process in real time.
- **`market_NN_plus_ultra/`** — an ambitious research stack aimed at building the "ultimate" trading agent with massive feature intake, deep temporal transformers, and risk-aware optimisation tooling.
- **`shared/`** — reusable UI styling and helpers consumed by both experiences.

Each project is fully static, so you can explore them by serving the repository root with any static file server (for example `npx serve .`).

## Building and launching the Asteroid Game client

The `asteroid-game` binary is built with Cargo. A release build produces a
native executable under `target/release/` that you can launch directly on the
platform you built it for.

```bash
cargo build -p asteroid-game --bin asteroid-game --release
```

After the build completes you will find:

| Platform | Binary location | How to launch |
| --- | --- | --- |
| **macOS** | `target/release/asteroid-game` | `open target/release/asteroid-game` or double-click it in Finder. |
| **Linux** | `target/release/asteroid-game` | `./target/release/asteroid-game` from a terminal. |
| **Windows** | `target\release\asteroid-game.exe` | Double-click the executable in Explorer or run it from PowerShell / Command Prompt. |

If you prefer a debug build while iterating on code, replace `--release` with
`--profile dev` or run `cargo run -p asteroid-game --bin asteroid-game`, which
builds and launches the executable in one step for the host platform.

### macOS GPU compatibility

On Apple GPUs prior to macOS 13 the Metal backend does not expose the
`atomic_compare_exchange` shader primitive, which causes the headless compute
pipeline to fail with an error similar to:

```
wgpu error: Validation Error
Caused by:
    In Device::create_compute_pipeline
      note: label = `headless-point-cull-pipeline`
    Internal error: MSL: FeatureNotImplemented("atomic CompareExchange")
```

Until we ship a Metal-friendly version of the pipeline, you can run the game in
the standard (non-headless) mode by omitting any `--headless` flag when you
launch the binary. When possible, updating macOS to Ventura (13) or newer also
unlocks the required Metal feature.

For more build and runtime notes, see [docs/troubleshooting.md](docs/troubleshooting.md).
