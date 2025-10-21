# Omni vs. Hybrid Backbone Benchmark

Generated: 2025-10-21 17:46 UTC

## Architecture Summary
| architecture | scenario_count | best_metric | best_metric_label | best_model_path | median_metric | mean_metric | median_duration | mean_duration | mean_profitability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_transformer | 3 | 32959.699219 | fx-hybrid | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/fx/plus-ultra-000-val_loss-0.0000.ckpt | 35167.488281 | 55806.572917 | 1.204321 | 1.26759 | -0.0001 |
| omni_mixture | 3 | 32986.566406 | fx-omni | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/fx/plus-ultra-000-val_loss-0.0000-v1.ckpt | 35199.253906 | 55770.463542 | 1.480337 | 1.483329 | 0.003557 |

## Per-Universe Leaders (val/loss)
| universe | rank | architecture | label | metric_val_loss | duration_seconds | best_model_path |
| --- | --- | --- | --- | --- | --- | --- |
| Crypto Majors | 1 | omni_mixture | crypto-omni | 99125.570312 | 1.34149 | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/crypto/plus-ultra-000-val_loss-0.0000-v1.ckpt |
| Equities Basket | 1 | hybrid_transformer | equities-hybrid | 35167.488281 | 1.546819 | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/equities/plus-ultra-000-val_loss-0.0000.ckpt |
| FX Crosses | 1 | hybrid_transformer | fx-hybrid | 32959.699219 | 1.204321 | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/fx/plus-ultra-000-val_loss-0.0000.ckpt |

## Per-Universe Leaders (profitability_roi)
| universe | rank | architecture | label | profitability_roi | duration_seconds | best_model_path |
| --- | --- | --- | --- | --- | --- | --- |
| Crypto Majors | 1 | omni_mixture | crypto-omni | 0.00659 | 1.34149 | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/crypto/plus-ultra-000-val_loss-0.0000-v1.ckpt |
| Equities Basket | 1 | omni_mixture | equities-omni | 0.005579 | 1.480337 | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/equities/plus-ultra-000-val_loss-0.0000-v1.ckpt |
| FX Crosses | 1 | omni_mixture | fx-omni | -0.001497 | 1.62816 | /workspace/NeuralDojo/docs/benchmarks/fixtures/checkpoints/fx/plus-ultra-000-val_loss-0.0000-v1.ckpt |