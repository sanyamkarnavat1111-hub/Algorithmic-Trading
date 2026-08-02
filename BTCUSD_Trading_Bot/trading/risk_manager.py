"""
trading/risk_manager.py — Position sizing and stop-loss calculation.

Rules (non-negotiable):
  - 2% of portfolio per trade (fixed)
  - Stop-loss = entry_price - (1.5 × ATR) for BUY
               entry_price + (1.5 × ATR) for SELL (short)
  - Max 1 open trade per model at a time
  - Daily loss circuit breaker: pause model if -5% in 24h
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from data.database import get_connection, log_event
from config import POSITION_SIZE_PCT, STOP_LOSS_ATR_MULT, MAX_OPEN_TRADES, DAILY_LOSS_LIMIT


def get_portfolio_value(model_id: str) -> float:
    """
    Calculate current portfolio value for a model.
    = starting capital + sum of all closed P&L
    Paper trading uses unlimited starting capital (1,000,000 USDT).
    """
    from config import PAPER_STARTING_CAPITAL
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(SUM(pnl), 0)
                FROM trades
                WHERE model_id = %s AND status = 'CLOSED'
            """, (model_id,))
            total_pnl = float(cur.fetchone()[0])
    finally:
        conn.close()

    return PAPER_STARTING_CAPITAL + total_pnl


def calculate_position_size(model_id: str, entry_price: float) -> float:
    """
    Calculate how many BTC units to buy/sell.
    Position value = 2% of portfolio.
    Units = position_value / entry_price.
    """
    portfolio_value = get_portfolio_value(model_id)
    position_value = portfolio_value * POSITION_SIZE_PCT
    units = position_value / entry_price
    return round(units, 6)


def calculate_stop_loss(signal: str, entry_price: float, atr: float) -> float:
    """
    Set stop-loss based on ATR.
    BUY:  stop = entry - (1.5 × ATR)   [price drops this far → exit]
    SELL: stop = entry + (1.5 × ATR)   [price rises this far → exit short]
    """
    offset = STOP_LOSS_ATR_MULT * atr
    if signal == "BUY":
        return round(entry_price - offset, 2)
    else:  # SELL (short)
        return round(entry_price + offset, 2)


def has_open_trade(model_id: str) -> bool:
    """Check if this model already has an open trade. Max 1 at a time."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM trades
                WHERE model_id = %s AND status = 'OPEN'
            """, (model_id,))
            count = cur.fetchone()[0]
    finally:
        conn.close()
    return count >= MAX_OPEN_TRADES


def is_circuit_breaker_active(model_id: str) -> bool:
    """
    Check if the daily loss circuit breaker is active.
    Pauses the model if it lost >5% of portfolio in the last 24 hours.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(SUM(pnl), 0)
                FROM trades
                WHERE model_id = %s
                  AND status = 'CLOSED'
                  AND closed_at > NOW() - INTERVAL '24 hours'
            """, (model_id,))
            daily_pnl = float(cur.fetchone()[0])
    finally:
        conn.close()

    portfolio_value = get_portfolio_value(model_id)
    daily_loss_pct = daily_pnl / portfolio_value if portfolio_value > 0 else 0

    if daily_loss_pct < -DAILY_LOSS_LIMIT:
        log_event("WARNING",
                  f"Circuit breaker active: daily loss = {daily_loss_pct:.2%}",
                  model_id=model_id)
        return True
    return False


def should_stop_loss_trigger(trade: dict, current_price: float) -> bool:
    """
    Check if the current price has hit the stop-loss for an open trade.
    BUY trade:  stop-loss triggers if price drops BELOW stop_loss
    SELL trade: stop-loss triggers if price rises ABOVE stop_loss
    """
    signal    = trade["signal"]
    stop_loss = float(trade["stop_loss"])

    if signal == "BUY"  and current_price <= stop_loss:
        return True
    if signal == "SELL" and current_price >= stop_loss:
        return True
    return False
