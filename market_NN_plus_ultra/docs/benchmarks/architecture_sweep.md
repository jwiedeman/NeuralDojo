# Architecture Sweep Benchmark Log

This log captures the queue configuration and telemetry status for the omni-scale,
hybrid, and state-space baseline sweeps orchestrated via
`scripts/benchmarks/architecture_sweep.py`.

## Queue Configuration

| Label | Config | Accelerator | Purpose |
| --- | --- | --- | --- |
| omni-gpu | `configs/benchmarks/omni_gpu.yaml` | GPU (bf16) | High-capacity omni-scale backbone profiling |
| hybrid-gpu | `configs/benchmarks/hybrid_gpu.yaml` | GPU (bf16) | Hybrid transformer comparison point |
| baseline-cpu | `configs/benchmarks/baseline_cpu.yaml` | CPU (fp32) | Lightweight regression + drift detector |

Each config trims epochs and batch coverage so sweeps can run quickly for smoke
testing while still emitting profiler telemetry.

## Profiler Trace Status

| Run | Throughput (samples/s) | VRAM (GB) | Latency (ms/step) | Status |
| --- | --- | --- | --- | --- |
| omni-gpu | _pending_ | _pending_ | _pending_ | Awaiting dedicated GPU allocation; CLI verified locally without execution. |
| hybrid-gpu | _pending_ | _pending_ | _pending_ | Queued via harness but blocked on GPU resources. |
| baseline-cpu | _pending_ | N/A | _pending_ | CPU run requires dataset fixture; skipped in CI due to resource limits. |

The current CI environment lacks the required accelerators and dataset volume to
emit profiler traces. Once hardware is available, execute the following commands
to populate the telemetry table above:

```bash
python scripts/benchmarks/architecture_sweep.py \
  --config configs/benchmarks/omni_gpu.yaml \
  --architectures omni_mixture \
  --model-dims 768,512 \
  --depths 12,16 \
  --horizons 5 \
  --output docs/benchmarks/omni_gpu.parquet

python scripts/benchmarks/architecture_sweep.py \
  --config configs/benchmarks/hybrid_gpu.yaml \
  --architectures hybrid_transformer \
  --model-dims 512,384 \
  --depths 8,10 \
  --horizons 5 \
  --output docs/benchmarks/hybrid_gpu.parquet

python scripts/benchmarks/architecture_sweep.py \
  --config configs/benchmarks/baseline_cpu.yaml \
  --architectures state_space \
  --model-dims 256 \
  --depths 6 \
  --horizons 5 \
  --accelerator cpu \
  --output docs/benchmarks/baseline_cpu.parquet
```

After collecting the parquet outputs, append throughput, VRAM, and latency
statistics to the table so downstream reporting jobs can ingest the telemetry.
