# Market NN Plus Ultra - Quick Start Guide

## Prerequisites

- Python 3.10+
- GPU with CUDA support (recommended) or CPU
- ~10GB disk space for data and checkpoints

## Setup (One-Time)

### 1. Install Dependencies

```bash
cd /home/user/NeuralDojo/market_NN_plus_ultra

# Install the package and all dependencies
pip install -e .
```

This installs:
- PyTorch (neural networks)
- PyTorch Lightning (training)
- pandas, numpy (data processing)
- yfinance (market data)
- ta (technical analysis)
- And other dependencies

### 2. Verify Installation

```bash
python scripts/launch.py --skip-checks
# Should show the startup banner
```

## Quick Start

### Option A: Automated Setup + Training (Recommended)

```bash
# Full setup: install deps + fetch data + start training
python scripts/launch.py --setup --mode swing
```

This will:
1. Install the package
2. Fetch 2 years of historical data for 100+ tickers
3. Start continuous RL training for swing trading

### Option B: Step-by-Step

```bash
# 1. Install package
pip install -e .

# 2. Fetch market data (takes ~10-15 minutes)
python scripts/fetch_market_data.py data/market.db --interval 1h --period 2y

# 3. Run single training cycle
python scripts/automation/retrain.py \
  --dataset data/market.db \
  --train-config configs/swing_trading.yaml \
  --run-reinforcement \
  --run-evaluation
```

## Trading Modes

### Swing Trading (Default)
- **Timeframe**: 1-hour candles
- **Holding period**: Days to weeks
- **Config**: `configs/swing_trading.yaml`

```bash
python scripts/launch.py --mode swing
```

### Day Trading
- **Timeframe**: 15-minute candles
- **Holding period**: Hours
- **Config**: `configs/day_trading.yaml`

```bash
python scripts/launch.py --mode day --interval 15m
```

## What's Being Trained

### Neural Network Architecture
The system uses an **Omni-Mixture** backbone combining:
- Cross-scale attention (captures patterns at multiple timeframes)
- State-space models (long-range memory)
- Dilated convolutions (local patterns)

### RL Training Pipeline
1. **Pretraining** (optional): Self-supervised learning on price patterns
2. **Supervised Training**: Learn to predict future returns
3. **PPO Fine-tuning**: Optimize for risk-adjusted returns (Sharpe, Sortino)

### Risk-Aware Rewards
The RL agent optimizes for:
- Sharpe ratio (risk-adjusted returns)
- Sortino ratio (downside risk)
- Drawdown minimization
- Tail risk (CVaR)

## Shadow Trading

**IMPORTANT**: This system only does paper trading (shadow trading). It will:
- Track hypothetical positions
- Calculate theoretical P&L
- Log all "trades" to SQLite
- Never execute real trades

Shadow portfolio state is saved to `data/shadow_portfolio.json`.

## Monitoring Training

### View Logs
```bash
tail -f continuous_training.log
```

### Check Shadow Performance
```python
from market_nn_plus_ultra.trading import ShadowPortfolio
portfolio = ShadowPortfolio.load_state(Path("data/shadow_portfolio.json"))
print(portfolio.get_performance_summary())
```

### View Training Results
```bash
cat outputs/evaluations/iteration_results.jsonl | python -m json.tool
```

## Ticker Universe

The system tracks 100+ tickers including:

| Category | Examples |
|----------|----------|
| Indices | SPY, QQQ, IWM, DIA |
| Tech | AAPL, MSFT, NVDA, AMD |
| Financials | JPM, GS, V, MA |
| Energy | XOM, CVX, OXY |
| Healthcare | JNJ, UNH, PFE |
| Sectors | XLK, XLF, XLE, XLV |
| Commodities | GLD, SLV, USO |
| International | EFA, EEM, FXI |

To use a subset:
```bash
python scripts/launch.py --categories mega_tech semis --mode day
```

## Configuration

### Adjust Model Size

Edit `configs/swing_trading.yaml`:
```yaml
model:
  model_dim: 512    # Reduce for faster training
  depth: 8          # Fewer layers
  heads: 8          # Attention heads
```

### Adjust RL Parameters

```yaml
reinforcement:
  total_updates: 500      # Number of PPO updates
  steps_per_rollout: 1024 # Samples per update
  learning_rate: 0.0001   # PPO learning rate
  risk_objective:
    sharpe_weight: 0.15   # Weight on Sharpe ratio
    drawdown_weight: 0.25 # Penalty for drawdowns
```

### Adjust Training Interval

```bash
# Train more frequently (every 15 minutes between iterations)
python scripts/launch.py --mode swing --cooldown 15

# Fetch new data every 2 hours
python scripts/continuous_training.py --fetch-interval 2
```

## Directory Structure

```
market_NN_plus_ultra/
├── data/
│   ├── market.db              # SQLite with OHLCV + features
│   └── shadow_portfolio.json  # Paper trading state
├── checkpoints/
│   ├── swing_trading/         # Saved models
│   └── continuous/            # Continuous training checkpoints
├── outputs/
│   └── evaluations/           # Training results
├── configs/
│   ├── swing_trading.yaml     # Multi-day config
│   └── day_trading.yaml       # Intraday config
└── scripts/
    ├── launch.py              # Main entry point
    ├── fetch_market_data.py   # Data fetcher
    └── continuous_training.py # Training daemon
```

## Stopping Training

Press `Ctrl+C` once for graceful shutdown. The daemon will:
1. Finish the current training step
2. Save checkpoint
3. Save shadow portfolio state
4. Exit cleanly

## Resuming Training

Just run the same command again:
```bash
python scripts/launch.py --mode swing
```

It will:
- Load existing data (or fetch new if stale)
- Continue from latest checkpoint
- Resume shadow portfolio state

## Troubleshooting

### Out of Memory
Reduce batch size in config:
```yaml
trainer:
  batch_size: 32  # Reduce from 64
```

### Training Too Slow
- Use GPU if available
- Reduce model size
- Increase `--cooldown` to train less frequently

### No GPU Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA PyTorch if needed
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Next Steps

1. **Monitor**: Watch the logs and shadow trading performance
2. **Tune**: Adjust risk weights based on observed behavior
3. **Evaluate**: Let it run for several cycles to see patterns
4. **Iterate**: Try different configs (day vs swing trading)

**Remember**: This is research/experimentation only. Never risk real money on untested models!
