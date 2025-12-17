"""Shadow Trading System for Market NN Plus Ultra.

This module provides paper trading / shadow trading functionality that tracks
hypothetical positions and P&L without executing real trades. It's designed for
validating model performance before any live deployment consideration.

IMPORTANT: This system is for SIMULATION ONLY. It does not execute real trades.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Position:
    """Represents a single position in the shadow portfolio."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    side: Literal["long", "short"]

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price * (1 if self.side == "long" else -1)

    @property
    def cost_basis(self) -> float:
        """Original cost of the position."""
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass(slots=True)
class Trade:
    """Record of a completed trade."""

    trade_id: int
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    slippage: float = 0.0
    signal_strength: float = 0.0
    model_confidence: float = 0.0


@dataclass
class ShadowPortfolio:
    """Paper trading portfolio that tracks positions and performance.

    This is for SHADOW TRADING ONLY - no real money is involved.
    """

    initial_capital: float = 100_000.0
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_history: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    # Transaction costs
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005  # 0.05% slippage

    # Risk limits
    max_position_pct: float = 0.10  # Max 10% per position
    max_total_exposure: float = 1.0  # Max 100% exposure
    max_drawdown_pct: float = 0.20  # Stop at 20% drawdown

    # State tracking
    peak_equity: float = field(init=False)
    trade_counter: int = 0
    is_halted: bool = False
    halt_reason: str = ""

    def __post_init__(self) -> None:
        self.cash = self.initial_capital
        self.peak_equity = self.initial_capital

    @property
    def total_equity(self) -> float:
        """Total portfolio value including cash and positions."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value

    @property
    def total_exposure(self) -> float:
        """Total exposure as fraction of equity."""
        if self.total_equity == 0:
            return 0.0
        gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        return gross_exposure / self.total_equity

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.total_equity) / self.peak_equity

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from closed trades."""
        return sum(
            t.price * t.quantity * (1 if t.side == "sell" else -1) - t.fees - t.slippage
            for t in self.trade_history
        )

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L from open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def _check_risk_limits(self) -> Tuple[bool, str]:
        """Check if risk limits are breached."""
        if self.current_drawdown >= self.max_drawdown_pct:
            return False, f"Max drawdown breached: {self.current_drawdown:.2%}"
        return True, ""

    def _calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate position size respecting risk limits."""
        max_position_value = self.total_equity * self.max_position_pct * abs(signal_strength)

        # Consider existing position
        existing = self.positions.get(symbol)
        if existing:
            current_value = abs(existing.market_value)
            available = max(0, max_position_value - current_value)
        else:
            available = max_position_value

        # Check total exposure limit
        current_exposure = self.total_exposure
        remaining_exposure = max(0, self.max_total_exposure - current_exposure)
        exposure_limit = remaining_exposure * self.total_equity

        available = min(available, exposure_limit, self.cash)

        if available <= 0:
            return 0.0

        return available / price

    def update_prices(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """Update position prices with latest market data."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

        # Update peak equity
        current_equity = self.total_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Record equity curve
        self.equity_curve.append((timestamp, current_equity))

        # Check risk limits
        ok, reason = self._check_risk_limits()
        if not ok:
            self.is_halted = True
            self.halt_reason = reason
            logger.warning(f"Shadow portfolio halted: {reason}")

    def execute_signal(
        self,
        symbol: str,
        signal: float,
        price: float,
        timestamp: datetime,
        confidence: float = 1.0,
    ) -> Optional[Trade]:
        """Execute a trading signal (paper trade only).

        Args:
            symbol: Ticker symbol
            signal: Trading signal (-1 to 1, negative = short, positive = long)
            price: Current market price
            timestamp: Signal timestamp
            confidence: Model confidence in the signal

        Returns:
            Trade record if executed, None otherwise
        """
        if self.is_halted:
            logger.warning(f"Cannot execute: portfolio halted ({self.halt_reason})")
            return None

        if abs(signal) < 0.1:
            return None

        # Determine trade direction
        side: Literal["long", "short"] = "long" if signal > 0 else "short"
        action: Literal["buy", "sell"] = "buy" if signal > 0 else "sell"

        existing = self.positions.get(symbol)

        # If we have an opposing position, close it first
        if existing and existing.side != side:
            close_trade = self._close_position(symbol, price, timestamp)
            if close_trade:
                self.trade_history.append(close_trade)

        # Calculate position size
        quantity = self._calculate_position_size(symbol, price, abs(signal))
        if quantity <= 0:
            return None

        # Apply transaction costs
        trade_value = quantity * price
        commission = trade_value * self.commission_rate
        slippage = trade_value * self.slippage_rate

        # Execute paper trade
        if action == "buy":
            adjusted_price = price * (1 + self.slippage_rate)
            total_cost = quantity * adjusted_price + commission
            if total_cost > self.cash:
                quantity = (self.cash - commission) / adjusted_price
                if quantity <= 0:
                    return None
                total_cost = quantity * adjusted_price + commission

            self.cash -= total_cost

            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                new_qty = pos.quantity + quantity
                pos.entry_price = (pos.entry_price * pos.quantity + adjusted_price * quantity) / new_qty
                pos.quantity = new_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=adjusted_price,
                    entry_time=timestamp,
                    current_price=price,
                    side=side,
                )
        else:
            adjusted_price = price * (1 - self.slippage_rate)
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=adjusted_price,
                entry_time=timestamp,
                current_price=price,
                side="short",
            )

        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=action,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=commission,
            slippage=slippage,
            signal_strength=signal,
            model_confidence=confidence,
        )

        self.trade_history.append(trade)
        logger.info(
            f"Shadow trade executed: {action.upper()} {quantity:.4f} {symbol} @ ${price:.2f} "
            f"(signal={signal:.3f}, confidence={confidence:.3f})"
        )

        return trade

    def _close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
    ) -> Optional[Trade]:
        """Close an existing position."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        quantity = pos.quantity

        # Apply transaction costs
        trade_value = quantity * price
        commission = trade_value * self.commission_rate
        slippage = trade_value * self.slippage_rate

        if pos.side == "long":
            adjusted_price = price * (1 - self.slippage_rate)
            proceeds = quantity * adjusted_price - commission
            self.cash += proceeds
            action: Literal["buy", "sell"] = "sell"
        else:
            adjusted_price = price * (1 + self.slippage_rate)
            pnl = (pos.entry_price - adjusted_price) * quantity - commission
            self.cash += pos.cost_basis + pnl
            action = "buy"

        del self.positions[symbol]

        self.trade_counter += 1
        return Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=action,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=commission,
            slippage=slippage,
            signal_strength=0.0,
            model_confidence=0.0,
        )

    def close_all_positions(self, prices: Dict[str, float], timestamp: datetime) -> List[Trade]:
        """Close all open positions."""
        trades = []
        for symbol in list(self.positions.keys()):
            price = prices.get(symbol, self.positions[symbol].current_price)
            trade = self._close_position(symbol, price, timestamp)
            if trade:
                trades.append(trade)
        return trades

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance metrics."""
        if not self.equity_curve:
            return {}

        equity_values = [e[1] for e in self.equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()

        # Basic metrics
        total_return = (self.total_equity - self.initial_capital) / self.initial_capital
        num_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.price > 0)

        # Risk metrics
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)
            sharpe = (returns.mean() * 252) / (volatility + 1e-8)
            sortino_returns = returns[returns < 0]
            downside_vol = sortino_returns.std() * np.sqrt(252) if len(sortino_returns) > 0 else 1e-8
            sortino = (returns.mean() * 252) / (downside_vol + 1e-8)

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = cumulative / rolling_max - 1
            max_drawdown = drawdowns.min()
        else:
            volatility = sharpe = sortino = max_drawdown = 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.total_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "num_trades": num_trades,
            "win_rate": winning_trades / num_trades if num_trades > 0 else 0,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "current_drawdown": self.current_drawdown,
            "total_exposure": self.total_exposure,
            "open_positions": len(self.positions),
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export trade history to DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()

        records = []
        for trade in self.trade_history:
            records.append({
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "price": trade.price,
                "fees": trade.fees,
                "slippage": trade.slippage,
                "signal_strength": trade.signal_strength,
                "model_confidence": trade.model_confidence,
            })
        return pd.DataFrame(records)

    def save_state(self, path: Path) -> None:
        """Save portfolio state to JSON."""
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "trade_counter": self.trade_counter,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "positions": {
                sym: {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat(),
                    "current_price": pos.current_price,
                    "side": pos.side,
                }
                for sym, pos in self.positions.items()
            },
            "trade_history": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "timestamp": t.timestamp.isoformat(),
                    "fees": t.fees,
                    "slippage": t.slippage,
                    "signal_strength": t.signal_strength,
                    "model_confidence": t.model_confidence,
                }
                for t in self.trade_history
            ],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq}
                for ts, eq in self.equity_curve
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: Path) -> "ShadowPortfolio":
        """Load portfolio state from JSON."""
        with open(path) as f:
            state = json.load(f)

        portfolio = cls(initial_capital=state["initial_capital"])
        portfolio.cash = state["cash"]
        portfolio.peak_equity = state["peak_equity"]
        portfolio.trade_counter = state["trade_counter"]
        portfolio.is_halted = state["is_halted"]
        portfolio.halt_reason = state["halt_reason"]

        for sym, pos_data in state["positions"].items():
            portfolio.positions[sym] = Position(
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                entry_price=pos_data["entry_price"],
                entry_time=datetime.fromisoformat(pos_data["entry_time"]),
                current_price=pos_data["current_price"],
                side=pos_data["side"],
            )

        for t_data in state["trade_history"]:
            portfolio.trade_history.append(Trade(
                trade_id=t_data["trade_id"],
                symbol=t_data["symbol"],
                side=t_data["side"],
                quantity=t_data["quantity"],
                price=t_data["price"],
                timestamp=datetime.fromisoformat(t_data["timestamp"]),
                fees=t_data["fees"],
                slippage=t_data["slippage"],
                signal_strength=t_data["signal_strength"],
                model_confidence=t_data["model_confidence"],
            ))

        for ec_data in state["equity_curve"]:
            portfolio.equity_curve.append((
                datetime.fromisoformat(ec_data["timestamp"]),
                ec_data["equity"],
            ))

        return portfolio


class ShadowTradingSession:
    """Manages a shadow trading session with a model."""

    def __init__(
        self,
        portfolio: ShadowPortfolio,
        db_path: Optional[Path] = None,
    ) -> None:
        self.portfolio = portfolio
        self.db_path = db_path
        self.session_start = datetime.now()

        if db_path:
            self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for trade logging."""
        if not self.db_path:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_trades (
                trade_id INTEGER PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                fees REAL,
                slippage REAL,
                signal_strength REAL,
                model_confidence REAL,
                portfolio_equity REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                equity REAL,
                cash REAL,
                positions_value REAL,
                drawdown REAL
            )
        """)
        conn.commit()
        conn.close()

    def log_trade(self, trade: Trade) -> None:
        """Log a trade to the database."""
        if not self.db_path:
            return

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO shadow_trades
            (trade_id, session_id, timestamp, symbol, side, quantity, price,
             fees, slippage, signal_strength, model_confidence, portfolio_equity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id,
            self.session_start.isoformat(),
            trade.timestamp.isoformat(),
            trade.symbol,
            trade.side,
            trade.quantity,
            trade.price,
            trade.fees,
            trade.slippage,
            trade.signal_strength,
            trade.model_confidence,
            self.portfolio.total_equity,
        ))
        conn.commit()
        conn.close()

    def log_equity(self, timestamp: datetime) -> None:
        """Log current equity to the database."""
        if not self.db_path:
            return

        positions_value = sum(pos.market_value for pos in self.portfolio.positions.values())

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO shadow_equity
            (session_id, timestamp, equity, cash, positions_value, drawdown)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            self.session_start.isoformat(),
            timestamp.isoformat(),
            self.portfolio.total_equity,
            self.portfolio.cash,
            positions_value,
            self.portfolio.current_drawdown,
        ))
        conn.commit()
        conn.close()

    def process_signals(
        self,
        signals: Dict[str, float],
        prices: Dict[str, float],
        timestamp: datetime,
        confidences: Optional[Dict[str, float]] = None,
    ) -> List[Trade]:
        """Process a batch of trading signals.

        Args:
            signals: Dict of symbol -> signal (-1 to 1)
            prices: Dict of symbol -> current price
            timestamp: Current timestamp
            confidences: Optional dict of symbol -> model confidence

        Returns:
            List of executed trades
        """
        # Update prices first
        self.portfolio.update_prices(prices, timestamp)

        if self.portfolio.is_halted:
            logger.warning(f"Portfolio halted: {self.portfolio.halt_reason}")
            return []

        trades = []
        confidences = confidences or {}

        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            confidence = confidences.get(symbol, 1.0)
            trade = self.portfolio.execute_signal(
                symbol=symbol,
                signal=signal,
                price=prices[symbol],
                timestamp=timestamp,
                confidence=confidence,
            )

            if trade:
                trades.append(trade)
                self.log_trade(trade)

        self.log_equity(timestamp)
        return trades


__all__ = [
    "Position",
    "Trade",
    "ShadowPortfolio",
    "ShadowTradingSession",
]
