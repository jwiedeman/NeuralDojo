#!/usr/bin/env bash
set -euo pipefail

# Default working directory for artefacts inside the container.
export PLUS_ULTRA_HOME="${PLUS_ULTRA_HOME:-/workspace}"
export DATA_DIR="${DATA_DIR:-${PLUS_ULTRA_HOME}/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PLUS_ULTRA_HOME}/checkpoints}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PLUS_ULTRA_HOME}/outputs}"

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$OUTPUT_DIR"
cd "$PLUS_ULTRA_HOME"

if [[ $# -eq 0 ]]; then
  exec bash
fi

case "$1" in
  train)
    shift
    exec python scripts/train.py "$@"
    ;;
  pretrain)
    shift
    exec python scripts/pretrain.py "$@"
    ;;
  agent|run-agent)
    shift
    exec python scripts/run_agent.py "$@"
    ;;
  service)
    shift
    if [[ $# -eq 0 ]]; then
      echo "Usage: service --config configs/default.yaml [--checkpoint path.ckpt]" >&2
    fi
    exec python scripts/service.py --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  shell|bash)
    shift || true
    exec bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
