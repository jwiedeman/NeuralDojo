# NeuralGo Experiments

This repository now hosts a small suite of browser-first neural playgrounds:

- **`neuralgo/`** — the original Go self-play lab, preserved exactly as before but now layered on top of the shared UI system.
- **`NeuralMatch/`** — a new synthetic match-3 evaluator that trains on aggressively augmented puzzle layouts, now drawing on
  NeuralArena-inspired pattern curricula with multi-layer perceptrons.
- **`shared/`** — reusable UI styling and helpers consumed by both experiences.

Each project is fully static, so you can explore them by serving the repository root with any static file server (for example `npx serve .`).
