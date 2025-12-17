#!/usr/bin/env python
"""Fetch comprehensive historical market data for many tickers and store in SQLite.

This script downloads OHLCV data from Yahoo Finance for a large universe of tickers
across multiple asset classes and timeframes, suitable for swing trading and day trading.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Comprehensive ticker universe for diverse market exposure
TICKER_UNIVERSE = {
    # Major US Indices ETFs
    "indices": [
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV",
    ],
    # Sector ETFs
    "sectors": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
        "XBI", "XOP", "XHB", "XRT",
    ],
    # Mega Cap Tech
    "mega_tech": [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
        "AMD", "INTC", "CRM", "ORCL", "ADBE", "NFLX",
    ],
    # Semiconductors
    "semis": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT", "LRCX", "KLAC",
        "MRVL", "ON", "TXN", "ADI", "MCHP",
    ],
    # Financials
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
        "V", "MA", "PYPL", "SQ", "COIN",
    ],
    # Healthcare/Biotech
    "healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
        "MRNA", "BIIB", "GILD", "REGN", "VRTX",
    ],
    # Energy
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
        "OXY", "DVN", "HAL", "BKR",
    ],
    # Consumer
    "consumer": [
        "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
        "DIS", "CMCSA", "ABNB", "BKNG",
    ],
    # Industrials
    "industrials": [
        "CAT", "DE", "BA", "HON", "UPS", "FDX", "LMT", "RTX", "GE", "MMM",
    ],
    # Volatility & Leveraged (for regime signals)
    "volatility": [
        "VXX", "UVXY", "SVXY", "SQQQ", "TQQQ", "SPXU", "UPRO",
    ],
    # Commodities ETFs
    "commodities": [
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC",
    ],
    # International ETFs
    "international": [
        "EFA", "EEM", "VEA", "VWO", "FXI", "EWJ", "EWZ", "EWG",
    ],
    # Bonds/Fixed Income
    "bonds": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "BND", "AGG", "TIP",
    ],
    # REITs
    "reits": [
        "VNQ", "IYR", "XLRE", "O", "AMT", "PLD", "EQIX", "SPG",
    ],
    # Crypto-adjacent (stocks exposed to crypto)
    "crypto_adjacent": [
        "COIN", "MSTR", "RIOT", "MARA", "HUT", "BITF",
    ],
}

# Flatten for default use
DEFAULT_TICKERS = []
for category, tickers in TICKER_UNIVERSE.items():
    DEFAULT_TICKERS.extend(tickers)
DEFAULT_TICKERS = list(dict.fromkeys(DEFAULT_TICKERS))  # Remove duplicates

# Intervals for different trading styles
INTERVALS = {
    "day_trading": ["1m", "5m", "15m", "30m", "1h"],
    "swing_trading": ["1h", "1d"],
    "position_trading": ["1d", "1wk"],
}


def get_max_period_for_interval(interval: str) -> str:
    """Return the maximum period allowed for a given interval by Yahoo Finance."""
    # Yahoo Finance data availability limits:
    # 1m: 7 days
    # 2m, 5m, 15m, 30m: 60 days
    # 1h: 730 days (~2 years)
    # 1d, 5d, 1wk, 1mo, 3mo: unlimited
    limits = {
        "1m": "7d",
        "2m": "60d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "60m": "2y",
        "1h": "2y",
        "1d": "max",
        "5d": "max",
        "1wk": "max",
        "1mo": "max",
        "3mo": "max",
    }
    return limits.get(interval, "max")


def fetch_ticker_data(
    ticker: str,
    interval: str = "1h",
    period: str = "auto",
    start: Optional[str] = None,
    end: Optional[str] = None,
    retries: int = 3,
    delay: float = 0.5,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a single ticker with retry logic."""
    import yfinance as yf

    # Auto-detect max period for interval
    if period == "auto" or period == "max":
        period = get_max_period_for_interval(interval)

    for attempt in range(retries):
        try:
            ticker_obj = yf.Ticker(ticker)

            if start and end:
                df = ticker_obj.history(start=start, end=end, interval=interval)
            else:
                df = ticker_obj.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {ticker} ({interval})")
                return None

            # Standardize column names
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            # Rename datetime/date column to timestamp
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})

            # Add symbol column
            df['symbol'] = ticker

            # Select and order columns
            required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in required_cols if c in df.columns]
            df = df[available_cols]

            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Remove timezone info for SQLite compatibility
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            logger.info(f"Fetched {len(df)} rows for {ticker} ({interval})")
            return df

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    logger.error(f"Failed to fetch {ticker} after {retries} attempts")
    return None


def fetch_all_tickers(
    tickers: list[str],
    interval: str = "1h",
    period: str = "auto",
    start: Optional[str] = None,
    end: Optional[str] = None,
    rate_limit_delay: float = 0.1,
) -> pd.DataFrame:
    """Fetch data for all tickers and combine into a single DataFrame."""
    all_data = []
    failed_tickers = []

    # Show actual period being used
    actual_period = period if period not in ("auto", "max") else get_max_period_for_interval(interval)
    logger.info(f"Using period '{actual_period}' for interval '{interval}'")

    for i, ticker in enumerate(tickers):
        logger.info(f"Fetching {ticker} ({i + 1}/{len(tickers)})")

        df = fetch_ticker_data(ticker, interval, period, start, end)
        if df is not None and not df.empty:
            all_data.append(df)
        else:
            failed_tickers.append(ticker)

        # Rate limiting
        if i < len(tickers) - 1:
            time.sleep(rate_limit_delay)

    if failed_tickers:
        logger.warning(f"Failed to fetch {len(failed_tickers)} tickers: {failed_tickers}")

    if not all_data:
        raise RuntimeError("No data fetched for any ticker")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

    logger.info(f"Combined dataset: {len(combined)} total rows, {combined['symbol'].nunique()} tickers")
    return combined


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional features for each symbol."""
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator, SMAIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

    result_frames = []

    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].copy().sort_values('timestamp')

        if len(sym_df) < 50:
            logger.warning(f"Skipping {symbol}: insufficient data ({len(sym_df)} rows)")
            continue

        try:
            # Price-based features
            sym_df['log_return'] = np.log(sym_df['close']).diff()
            sym_df['pct_return'] = sym_df['close'].pct_change()

            # Moving averages
            sym_df['sma_20'] = SMAIndicator(sym_df['close'], window=20).sma_indicator()
            sym_df['sma_50'] = SMAIndicator(sym_df['close'], window=50).sma_indicator()
            sym_df['ema_12'] = EMAIndicator(sym_df['close'], window=12).ema_indicator()
            sym_df['ema_26'] = EMAIndicator(sym_df['close'], window=26).ema_indicator()

            # Momentum
            sym_df['rsi_14'] = RSIIndicator(sym_df['close'], window=14).rsi()
            macd = MACD(sym_df['close'])
            sym_df['macd'] = macd.macd()
            sym_df['macd_signal'] = macd.macd_signal()
            sym_df['macd_hist'] = macd.macd_diff()

            # Volatility
            bb = BollingerBands(sym_df['close'], window=20)
            sym_df['bb_upper'] = bb.bollinger_hband()
            sym_df['bb_lower'] = bb.bollinger_lband()
            sym_df['bb_width'] = bb.bollinger_wband()
            sym_df['atr_14'] = AverageTrueRange(sym_df['high'], sym_df['low'], sym_df['close'], window=14).average_true_range()

            # Volume
            sym_df['obv'] = OnBalanceVolumeIndicator(sym_df['close'], sym_df['volume']).on_balance_volume()
            sym_df['volume_sma_20'] = sym_df['volume'].rolling(20).mean()
            sym_df['volume_ratio'] = sym_df['volume'] / sym_df['volume_sma_20']

            # Realized volatility
            sym_df['realized_vol_20'] = sym_df['log_return'].rolling(20).std() * np.sqrt(252)

            # Price distance from moving averages
            sym_df['price_to_sma20'] = sym_df['close'] / sym_df['sma_20']
            sym_df['price_to_sma50'] = sym_df['close'] / sym_df['sma_50']

            # Higher moments
            sym_df['rolling_skew_20'] = sym_df['log_return'].rolling(20).skew()
            sym_df['rolling_kurt_20'] = sym_df['log_return'].rolling(20).kurt()

            result_frames.append(sym_df)

        except Exception as e:
            logger.warning(f"Feature computation failed for {symbol}: {e}")
            result_frames.append(sym_df)

    if not result_frames:
        return df

    result = pd.concat(result_frames, ignore_index=True)
    return result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)


def generate_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate regime labels for each symbol."""
    regimes = []

    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].copy().sort_values('timestamp')

        if len(sym_df) < 100:
            continue

        # Volatility regime (low/medium/high)
        if 'realized_vol_20' in sym_df.columns:
            vol = sym_df['realized_vol_20'].dropna()
            if len(vol) > 0:
                vol_q33 = vol.quantile(0.33)
                vol_q66 = vol.quantile(0.66)

                for _, row in sym_df.iterrows():
                    if pd.notna(row.get('realized_vol_20')):
                        if row['realized_vol_20'] <= vol_q33:
                            vol_regime = 0  # Low vol
                        elif row['realized_vol_20'] <= vol_q66:
                            vol_regime = 1  # Medium vol
                        else:
                            vol_regime = 2  # High vol

                        regimes.append({
                            'timestamp': row['timestamp'],
                            'symbol': symbol,
                            'name': 'volatility_regime',
                            'value': vol_regime,
                        })

        # Trend regime (bearish/neutral/bullish)
        if 'sma_20' in sym_df.columns and 'sma_50' in sym_df.columns:
            for _, row in sym_df.iterrows():
                if pd.notna(row.get('sma_20')) and pd.notna(row.get('sma_50')):
                    if row['close'] > row['sma_20'] > row['sma_50']:
                        trend = 2  # Bullish
                    elif row['close'] < row['sma_20'] < row['sma_50']:
                        trend = 0  # Bearish
                    else:
                        trend = 1  # Neutral

                    regimes.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'name': 'trend_regime',
                        'value': trend,
                    })

    if not regimes:
        return pd.DataFrame(columns=['timestamp', 'symbol', 'name', 'value'])

    return pd.DataFrame(regimes)


def write_to_sqlite(
    df: pd.DataFrame,
    db_path: Path,
    regimes_df: Optional[pd.DataFrame] = None,
) -> None:
    """Write data to SQLite database with proper schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    try:
        # Create assets table
        symbols = df['symbol'].unique()
        assets_df = pd.DataFrame({
            'symbol': symbols,
            'name': symbols,
            'asset_class': ['equity'] * len(symbols),
        })
        assets_df.to_sql('assets', conn, if_exists='replace', index=False)

        # Create series table (main OHLCV + features)
        df.to_sql('series', conn, if_exists='replace', index=False)

        # Create regimes table
        if regimes_df is not None and not regimes_df.empty:
            regimes_df.to_sql('regimes', conn, if_exists='replace', index=False)

        # Create indices for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_series_timestamp ON series(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_series_symbol ON series(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_series_ts_sym ON series(timestamp, symbol)")

        if regimes_df is not None and not regimes_df.empty:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_regimes_ts_sym ON regimes(timestamp, symbol)")

        conn.commit()

        # Print summary
        cursor = conn.execute("SELECT COUNT(*) FROM series")
        row_count = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM series")
        symbol_count = cursor.fetchone()[0]

        logger.info(f"Database written to {db_path}")
        logger.info(f"  - {row_count:,} total rows")
        logger.info(f"  - {symbol_count} unique symbols")

    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical market data for neural network training",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output SQLite database",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to fetch (default: comprehensive universe)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(TICKER_UNIVERSE.keys()),
        help="Ticker categories to include",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "1d", "1wk"],
        help="Data interval (default: 1h)",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="Data period (e.g., '1y', '2y', 'max')",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--compute-features",
        action="store_true",
        default=True,
        help="Compute technical indicators (default: True)",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip feature computation",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Delay between ticker fetches in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine tickers to fetch
    if args.tickers:
        tickers = args.tickers
    elif args.categories:
        tickers = []
        for cat in args.categories:
            tickers.extend(TICKER_UNIVERSE.get(cat, []))
        tickers = list(dict.fromkeys(tickers))
    else:
        tickers = DEFAULT_TICKERS

    logger.info(f"Fetching data for {len(tickers)} tickers")
    logger.info(f"Interval: {args.interval}, Period: {args.period}")

    # Fetch data
    df = fetch_all_tickers(
        tickers,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
        rate_limit_delay=args.rate_limit,
    )

    # Compute features
    if args.compute_features and not args.no_features:
        logger.info("Computing technical features...")
        df = compute_features(df)

    # Generate regime labels
    logger.info("Generating regime labels...")
    regimes_df = generate_regime_labels(df)

    # Write to SQLite
    write_to_sqlite(df, args.output, regimes_df)

    logger.info("Data fetch complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
