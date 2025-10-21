# GPU Containerisation Guide

This guide explains how to build and run the Market NN Plus Ultra container
image that targets GPU-equipped hosts. The image ships with CUDA-enabled
PyTorch, Lightning, and the complete project stack so you can launch
training, pretraining, inference agents, or the FastAPI service without
needing to manage Python environments manually.

## Build the image

```bash
cd market_NN_plus_ultra
# Use docker buildx for consistent multi-architecture builds.
docker buildx build \
  -f docker/Dockerfile.gpu \
  -t plus-ultra:latest \
  .
```

> **Tip:** The build context honours `.dockerignore`, so local artefacts such as
> checkpoints, fixtures, or notebooks stay out of the image. Set
> `--build-arg PIP_EXTRA_INDEX_URL=...` if you host private wheels.

## Shared volume layout

The container expects (and will create when missing) the following directories:

| Path inside container | Purpose |
| --------------------- | ------- |
| `/workspace/data`     | SQLite fixtures and cached market data. |
| `/workspace/checkpoints` | Lightning and PPO checkpoints. |
| `/workspace/outputs`  | Prediction exports, benchmark reports, diagnostics. |

Mount host directories into these paths to persist artefacts across runs:

```bash
docker run --gpus all \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/outputs:/workspace/outputs" \
  plus-ultra:latest bash
```

## Entrypoint commands

The image exposes a multi-command entrypoint to streamline common workflows.
Pass the desired sub-command followed by the original CLI arguments:

### Supervised training

```bash
docker run --gpus all --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  plus-ultra:latest \
  train --config configs/default.yaml --accelerator gpu --devices 1
```

### Self-supervised pretraining

```bash
docker run --gpus all --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  plus-ultra:latest \
  pretrain --config configs/pretrain.yaml --accelerator gpu --devices 1
```

### Offline inference agent

```bash
docker run --gpus all --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/outputs:/workspace/outputs" \
  plus-ultra:latest \
  agent --config configs/default.yaml \
  --checkpoint checkpoints/default/last.ckpt \
  --device cuda:0 \
  --output outputs/predictions.parquet
```

### FastAPI inference service

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  plus-ultra:latest \
  service --config configs/default.yaml --checkpoint checkpoints/default/last.ckpt
```

Set `PORT` in the environment to override the default exposed port.

## Troubleshooting

* **`torch.cuda.is_available()` returns `False`** – ensure the host has the
  NVIDIA Container Toolkit installed and pass `--gpus all` (or an explicit
  device list) to `docker run`.
* **Permission errors on mounted volumes** – run containers with matching user
  IDs (`-u $(id -u):$(id -g)`) or relax permissions on the host directories.
* **Slow builds due to wheel compilation** – pre-build wheels in a volume and
  mount them, or supply extra indices via `PIP_EXTRA_INDEX_URL`.

With the container in place, orchestration jobs and researchers share a single
reproducible runtime that mirrors the optimisation environment used in the
implementation plan.
