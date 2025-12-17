#!/usr/bin/env python
"""Launch script for Market NN Plus Ultra.

This script handles:
1. Dependency verification
2. Initial data fetching
3. Training mode selection (swing/day trading)
4. Starting continuous training or single runs

Usage:
    # First time setup + launch continuous training
    python scripts/launch.py --setup --mode swing

    # Quick start with existing data
    python scripts/launch.py --mode day

    # Single training run (no daemon)
    python scripts/launch.py --mode swing --single-run

    # Just fetch data
    python scripts/launch.py --fetch-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import os


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗          ║
║     ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║          ║
║     ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║          ║
║     ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║          ║
║     ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗     ║
║     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝     ║
║                                                              ║
║              MARKET NN PLUS ULTRA                            ║
║         Continuous RL Trading System                         ║
║                                                              ║
║  ⚠️  SHADOW TRADING ONLY - NO REAL MONEY                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_python_version():
    """Verify Python version."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        print(f"   Current: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    required = [
        ("torch", "PyTorch"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("yfinance", "Yahoo Finance"),
        ("ta", "Technical Analysis"),
        ("sqlalchemy", "SQLAlchemy"),
        ("einops", "Einops"),
    ]

    all_ok = True
    missing = []

    for package, name in required:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (pip install {package})")
            missing.append(package)
            all_ok = False

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_name}")
        else:
            print("⚠️  No CUDA GPU detected (will use CPU)")
    except Exception as e:
        print(f"⚠️  GPU check failed: {e}")

    return all_ok, missing


def install_package():
    """Install the market_nn_plus_ultra package."""
    project_root = Path(__file__).parent.parent

    print("\n📦 Installing Market NN Plus Ultra package...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(project_root)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Installation failed:\n{result.stderr}")
        return False

    print("✅ Package installed")
    return True


def fetch_data(interval: str = "1h", categories: list[str] | None = None):
    """Fetch market data."""
    print("\n📊 Fetching market data...")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "fetch_market_data.py"),
        "data/market.db",
        "--interval", interval,
        "--period", "auto",
        "--compute-features",
    ]

    if categories:
        cmd.extend(["--categories"] + categories)

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print("❌ Data fetch failed")
        return False

    print("✅ Data fetched successfully")
    return True


def run_single_training(config_path: Path, device: str = "auto"):
    """Run a single training cycle."""
    print(f"\n🏋️ Starting single training run with {config_path.name}...")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "automation" / "retrain.py"),
        "--dataset", "data/market.db",
        "--train-config", str(config_path),
        "--run-reinforcement",
        "--run-evaluation",
        "--warm-start", "training",
    ]

    subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def run_continuous_training(config_path: Path, device: str = "auto", cooldown: float = 30.0):
    """Start the continuous training daemon."""
    print(f"\n🔄 Starting continuous training daemon with {config_path.name}...")
    print("   Press Ctrl+C to stop gracefully")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "continuous_training.py"),
        "--config", str(config_path),
        "--device", device,
        "--cooldown", str(cooldown),
    ]

    subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Market NN Plus Ultra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time setup with swing trading
  python scripts/launch.py --setup --mode swing

  # Day trading with quick iterations
  python scripts/launch.py --mode day --cooldown 15

  # Single training run (no continuous loop)
  python scripts/launch.py --mode swing --single-run

  # Just fetch fresh data
  python scripts/launch.py --fetch-only --interval 15m

Trading Modes:
  swing  - Multi-day holds, 1h candles, longer horizons
  day    - Intraday, 15m candles, shorter horizons
        """,
    )

    parser.add_argument("--setup", action="store_true",
                        help="Run first-time setup (install + fetch data)")
    parser.add_argument("--mode", choices=["swing", "day"], default="swing",
                        help="Trading mode (default: swing)")
    parser.add_argument("--single-run", action="store_true",
                        help="Run single training cycle instead of daemon")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch data, don't train")
    parser.add_argument("--interval", default=None,
                        choices=["1m", "5m", "15m", "30m", "1h", "1d"],
                        help="Data interval (default: based on mode)")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--cooldown", type=float, default=30.0,
                        help="Minutes between training iterations")
    parser.add_argument("--categories", nargs="+",
                        help="Ticker categories to fetch")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip dependency checks")

    args = parser.parse_args()

    print_banner()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Dependency checks
    if not args.skip_checks:
        print("\n🔍 Checking dependencies...\n")

        if not check_python_version():
            return 1

        deps_ok, missing = check_dependencies()

        if not deps_ok:
            print(f"\n❌ Missing dependencies: {', '.join(missing)}")
            print("   Run: pip install -e .")
            if not args.setup:
                return 1

    # Setup mode
    if args.setup:
        if not install_package():
            return 1

    # Determine data interval based on mode
    if args.interval:
        interval = args.interval
    elif args.mode == "day":
        interval = "15m"
    else:
        interval = "1h"

    # Fetch data if needed
    data_path = project_root / "data" / "market.db"
    if args.setup or args.fetch_only or not data_path.exists():
        if not fetch_data(interval, args.categories):
            return 1

    if args.fetch_only:
        print("\n✅ Data fetch complete!")
        return 0

    # Select config
    if args.mode == "day":
        config_path = project_root / "configs" / "day_trading.yaml"
    else:
        config_path = project_root / "configs" / "swing_trading.yaml"

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1

    # Run training
    if args.single_run:
        run_single_training(config_path, args.device)
    else:
        run_continuous_training(config_path, args.device, args.cooldown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
