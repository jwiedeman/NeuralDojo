# NeuralGo Experiments

This repository now hosts a small suite of browser-first neural playgrounds:

- **`neuralgo/`** — the original Go self-play lab, preserved exactly as before but now layered on top of the shared UI system.
- **`NeuralMatch/`** — a new synthetic match-3 evaluator that trains on aggressively augmented puzzle layouts.
- **`NeuralStocks/`** — a market forecaster playground that streams SPY closes through a compact MLP and visualises the learning process in real time.
- **`shared/`** — reusable UI styling and helpers consumed by both experiences.

Each project is fully static, so you can explore them by serving the repository root with any static file server (for example `npx serve .`).
