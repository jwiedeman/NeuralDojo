#!/usr/bin/env python
"""Continuous RL Training Daemon for Market NN Plus Ultra.

This script runs continuous reinforcement learning training, periodically
fetching new market data, training the model, and running shadow trading
evaluation.

Usage:
    python scripts/continuous_training.py --config configs/swing_trading.yaml

The daemon will:
1. Fetch latest market data
2. Run pretraining (if enabled)
3. Run supervised training
4. Run PPO fine-tuning
5. Evaluate on shadow trading
6. Loop back to step 1
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('continuous_training.log'),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for the continuous training daemon."""

    # Data
    db_path: Path = Path("data/market.db")
    data_fetch_interval_hours: float = 6.0
    data_interval: str = "1h"
    ticker_categories: list[str] | None = None

    # Training
    training_config: Path = Path("configs/swing_trading.yaml")
    pretrain_config: Optional[Path] = None
    run_pretraining: bool = False
    run_supervised: bool = True
    run_reinforcement: bool = True

    # Evaluation
    run_shadow_trading: bool = True
    shadow_portfolio_path: Path = Path("data/shadow_portfolio.json")
    evaluation_output: Path = Path("outputs/evaluations")

    # Checkpoints
    checkpoint_dir: Path = Path("checkpoints/continuous")
    keep_n_checkpoints: int = 5

    # Daemon control
    max_iterations: int = 0  # 0 = infinite
    cooldown_minutes: float = 30.0
    device: str = "auto"


class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM."""

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        logger.info("Received shutdown signal, finishing current iteration...")
        self.kill_now = True


def fetch_latest_data(config: DaemonConfig) -> bool:
    """Fetch latest market data."""
    logger.info("Fetching latest market data...")

    try:
        from scripts.fetch_market_data import (
            fetch_all_tickers,
            compute_features,
            generate_regime_labels,
            write_to_sqlite,
            DEFAULT_TICKERS,
            TICKER_UNIVERSE,
        )

        # Determine tickers
        if config.ticker_categories:
            tickers = []
            for cat in config.ticker_categories:
                tickers.extend(TICKER_UNIVERSE.get(cat, []))
            tickers = list(dict.fromkeys(tickers))
        else:
            tickers = DEFAULT_TICKERS

        # Fetch data
        df = fetch_all_tickers(
            tickers,
            interval=config.data_interval,
            period="2y",  # Get 2 years of data
            rate_limit_delay=0.1,
        )

        if df.empty:
            logger.error("No data fetched")
            return False

        # Compute features
        logger.info("Computing features...")
        df = compute_features(df)

        # Generate regimes
        logger.info("Generating regime labels...")
        regimes_df = generate_regime_labels(df)

        # Write to SQLite
        write_to_sqlite(df, config.db_path, regimes_df)

        logger.info(f"Data fetch complete: {len(df)} rows, {df['symbol'].nunique()} symbols")
        return True

    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return False


def run_training_cycle(config: DaemonConfig, iteration: int) -> dict:
    """Run a complete training cycle."""
    results = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "pretraining": None,
        "supervised": None,
        "reinforcement": None,
        "shadow_trading": None,
    }

    try:
        from market_nn_plus_ultra.training import (
            load_experiment_from_file,
            run_training,
            run_reinforcement_finetuning,
            ReinforcementConfig,
        )
        from market_nn_plus_ultra.trading.shadow_trader import (
            ShadowPortfolio,
            ShadowTradingSession,
        )

        # Load experiment config
        experiment = load_experiment_from_file(config.training_config)

        # Update paths
        experiment.data.sqlite_path = str(config.db_path)
        experiment.trainer.checkpoint_dir = str(config.checkpoint_dir / f"iter_{iteration:04d}")

        # Determine device
        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device

        logger.info(f"Using device: {device}")

        checkpoint_path = None

        # Run pretraining if enabled
        if config.run_pretraining and config.pretrain_config:
            logger.info("Starting pretraining...")
            pretrain_exp = load_experiment_from_file(config.pretrain_config)
            pretrain_exp.data.sqlite_path = str(config.db_path)

            from market_nn_plus_ultra.training import run_pretraining
            pretrain_result = run_pretraining(pretrain_exp)
            results["pretraining"] = {
                "epochs": pretrain_result.epochs_completed,
                "final_loss": pretrain_result.final_loss,
            }
            checkpoint_path = pretrain_result.best_checkpoint_path

        # Run supervised training
        if config.run_supervised:
            logger.info("Starting supervised training...")
            training_result = run_training(
                experiment,
                pretrain_checkpoint_path=checkpoint_path,
            )
            results["supervised"] = {
                "epochs": training_result.epochs_completed,
                "best_val_loss": training_result.best_val_loss,
                "checkpoint": str(training_result.best_checkpoint_path),
            }
            checkpoint_path = training_result.best_checkpoint_path

        # Run reinforcement learning
        if config.run_reinforcement:
            logger.info("Starting PPO fine-tuning...")

            reinforcement_config = experiment.reinforcement or ReinforcementConfig()

            rl_result = run_reinforcement_finetuning(
                experiment,
                reinforcement_config=reinforcement_config,
                checkpoint_path=checkpoint_path,
                device=device,
            )

            results["reinforcement"] = {
                "updates": len(rl_result.updates),
                "final_reward": rl_result.updates[-1].mean_reward if rl_result.updates else 0,
                "eval_metrics": rl_result.evaluation_metrics,
            }

            # Save RL checkpoint
            rl_checkpoint_path = config.checkpoint_dir / f"iter_{iteration:04d}" / "rl_final.pt"
            rl_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy_state_dict": rl_result.policy_state_dict,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
            }, rl_checkpoint_path)

        # Run shadow trading evaluation
        if config.run_shadow_trading and checkpoint_path:
            logger.info("Running shadow trading evaluation...")
            shadow_results = run_shadow_evaluation(config, checkpoint_path, experiment)
            results["shadow_trading"] = shadow_results

        return results

    except Exception as e:
        logger.error(f"Training cycle failed: {e}", exc_info=True)
        results["error"] = str(e)
        return results


def run_shadow_evaluation(config: DaemonConfig, checkpoint_path: Path, experiment) -> dict:
    """Run shadow trading evaluation."""
    try:
        from market_nn_plus_ultra.trading.shadow_trader import (
            ShadowPortfolio,
            ShadowTradingSession,
        )
        from market_nn_plus_ultra.trading.agent import MarketNNPlusUltraAgent
        import pandas as pd

        # Load or create shadow portfolio
        if config.shadow_portfolio_path.exists():
            portfolio = ShadowPortfolio.load_state(config.shadow_portfolio_path)
            logger.info(f"Loaded existing portfolio: ${portfolio.total_equity:,.2f}")
        else:
            portfolio = ShadowPortfolio(initial_capital=100_000.0)
            logger.info("Created new shadow portfolio")

        # Create trading session
        session = ShadowTradingSession(
            portfolio=portfolio,
            db_path=config.evaluation_output / "shadow_trades.db",
        )

        # Load agent
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = MarketNNPlusUltraAgent(
            experiment_config=experiment,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # Run inference and get predictions
        agent_result = agent.run(evaluate=True)
        predictions = agent_result.predictions

        if predictions is None or predictions.empty:
            logger.warning("No predictions generated")
            return {"error": "No predictions"}

        # Get latest prices
        latest = predictions.groupby("symbol").last()

        # Generate signals from predictions
        signals = {}
        prices = {}
        for symbol in latest.index:
            row = latest.loc[symbol]
            if "prediction" in row:
                # Normalize prediction to signal
                pred = row["prediction"]
                signal = float(max(-1, min(1, pred)))
                signals[symbol] = signal
            if "close" in row:
                prices[symbol] = float(row["close"])

        # Execute signals
        timestamp = datetime.now()
        trades = session.process_signals(signals, prices, timestamp)

        # Save portfolio state
        config.shadow_portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        portfolio.save_state(config.shadow_portfolio_path)

        # Get performance summary
        perf = portfolio.get_performance_summary()

        logger.info(f"Shadow trading: {len(trades)} trades, equity=${perf.get('final_equity', 0):,.2f}")

        return {
            "trades_executed": len(trades),
            "performance": perf,
        }

    except Exception as e:
        logger.error(f"Shadow evaluation failed: {e}", exc_info=True)
        return {"error": str(e)}


def cleanup_old_checkpoints(config: DaemonConfig) -> None:
    """Remove old checkpoints, keeping only the most recent N."""
    if not config.checkpoint_dir.exists():
        return

    iteration_dirs = sorted(
        [d for d in config.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda x: x.name,
        reverse=True,
    )

    for old_dir in iteration_dirs[config.keep_n_checkpoints:]:
        logger.info(f"Removing old checkpoint: {old_dir}")
        import shutil
        shutil.rmtree(old_dir)


def save_iteration_results(config: DaemonConfig, results: dict) -> None:
    """Save iteration results to JSON."""
    output_path = config.evaluation_output / "iteration_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as f:
        f.write(json.dumps(results) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous RL Training Daemon")

    parser.add_argument("--config", type=Path, default=Path("configs/swing_trading.yaml"),
                        help="Training config file")
    parser.add_argument("--db-path", type=Path, default=Path("data/market.db"),
                        help="SQLite database path")
    parser.add_argument("--data-interval", default="1h",
                        choices=["15m", "30m", "1h", "1d"],
                        help="Data interval")
    parser.add_argument("--fetch-interval", type=float, default=6.0,
                        help="Hours between data fetches")
    parser.add_argument("--no-pretraining", action="store_true",
                        help="Skip pretraining")
    parser.add_argument("--no-supervised", action="store_true",
                        help="Skip supervised training")
    parser.add_argument("--no-reinforcement", action="store_true",
                        help="Skip PPO fine-tuning")
    parser.add_argument("--no-shadow-trading", action="store_true",
                        help="Skip shadow trading evaluation")
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Max iterations (0 = infinite)")
    parser.add_argument("--cooldown", type=float, default=30.0,
                        help="Minutes between iterations")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--categories", nargs="+",
                        help="Ticker categories to fetch")

    args = parser.parse_args()

    config = DaemonConfig(
        db_path=args.db_path,
        data_fetch_interval_hours=args.fetch_interval,
        data_interval=args.data_interval,
        ticker_categories=args.categories,
        training_config=args.config,
        run_pretraining=not args.no_pretraining,
        run_supervised=not args.no_supervised,
        run_reinforcement=not args.no_reinforcement,
        run_shadow_trading=not args.no_shadow_trading,
        max_iterations=args.max_iterations,
        cooldown_minutes=args.cooldown,
        device=args.device,
    )

    logger.info("=" * 60)
    logger.info("Market NN Plus Ultra - Continuous Training Daemon")
    logger.info("=" * 60)
    logger.info(f"Config: {config.training_config}")
    logger.info(f"Database: {config.db_path}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Data interval: {config.data_interval}")
    logger.info(f"Fetch interval: {config.data_fetch_interval_hours}h")
    logger.info(f"Cooldown: {config.cooldown_minutes}min")
    logger.info("=" * 60)

    killer = GracefulKiller()
    iteration = 0
    last_fetch = datetime.min

    while not killer.kill_now:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")

        # Check if we need to fetch new data
        time_since_fetch = datetime.now() - last_fetch
        if time_since_fetch > timedelta(hours=config.data_fetch_interval_hours):
            success = fetch_latest_data(config)
            if success:
                last_fetch = datetime.now()
            else:
                logger.warning("Data fetch failed, using existing data")

        # Run training cycle
        results = run_training_cycle(config, iteration)
        save_iteration_results(config, results)

        # Cleanup old checkpoints
        cleanup_old_checkpoints(config)

        # Log summary
        logger.info(f"\nIteration {iteration} complete")
        if results.get("reinforcement"):
            rl = results["reinforcement"]
            logger.info(f"  RL updates: {rl.get('updates', 0)}")
            logger.info(f"  Final reward: {rl.get('final_reward', 0):.6f}")
        if results.get("shadow_trading"):
            st = results["shadow_trading"]
            perf = st.get("performance", {})
            logger.info(f"  Shadow equity: ${perf.get('final_equity', 0):,.2f}")
            logger.info(f"  Sharpe ratio: {perf.get('sharpe_ratio', 0):.3f}")

        # Check iteration limit
        if config.max_iterations > 0 and iteration >= config.max_iterations:
            logger.info(f"Reached max iterations ({config.max_iterations}), stopping")
            break

        # Cooldown
        if not killer.kill_now:
            logger.info(f"Cooling down for {config.cooldown_minutes} minutes...")
            cooldown_end = datetime.now() + timedelta(minutes=config.cooldown_minutes)
            while datetime.now() < cooldown_end and not killer.kill_now:
                time.sleep(10)

    logger.info("\nDaemon shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
