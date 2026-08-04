"""
trading/portfolio_manager.py — Manages USDT and BTC balances.

Simple spot trading logic:
  - Buy BTC: spend USDT, receive BTC
  - Sell BTC: spend BTC, receive USDT
  - Cannot sell more BTC than held (no shorting)
  - Multiple positions allowed (can buy multiple times)
  - Tracks average buy price for P&L calculation

All state is stored in PostgreSQL (survives restarts).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_connection, log_event
from config import STARTING_USDT_BALANCE


MODEL_ID = "ai_15m"


def _ensure_portfolio_exists():
    """Create portfolio row if it doesn't exist yet."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolio WHERE model_id = %s", (MODEL_ID,))
            if cur.fetchone() is None:
                cur.execute("""
                    INSERT INTO portfolio (model_id, usdt_balance, btc_quantity, btc_avg_price)
                    VALUES (%s, %s, 0, 0)
                """, (MODEL_ID, STARTING_USDT_BALANCE))
        conn.commit()
    finally:
        conn.close()


def get_portfolio() -> dict:
    """
    Get current portfolio state.

    Returns:
      {
        "usdt_balance": float,
        "btc_quantity": float,
        "btc_avg_price": float,
        "total_value_usdt": float (at current price — caller must provide)
      }
    """
    _ensure_portfolio_exists()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT usdt_balance, btc_quantity, btc_avg_price
                FROM portfolio WHERE model_id = %s
            """, (MODEL_ID,))
            row = cur.fetchone()
    finally:
        conn.close()

    return {
        "usdt_balance": float(row[0]),
        "btc_quantity": float(row[1]),
        "btc_avg_price": float(row[2]),
    }


def buy_btc(amount_usdt: float, current_price: float) -> dict:
    """
    Buy BTC with USDT.

    Args:
        amount_usdt: how much USDT to spend (e.g., 150)
        current_price: current BTC price (e.g., 63000)

    Returns:
        Trade record dict, or None if insufficient balance.
    """
    _ensure_portfolio_exists()

    btc_to_buy = amount_usdt / current_price

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Get current portfolio
            cur.execute("""
                SELECT usdt_balance, btc_quantity, btc_avg_price
                FROM portfolio WHERE model_id = %s FOR UPDATE
            """, (MODEL_ID,))
            row = cur.fetchone()
            usdt_balance = float(row[0])
            btc_quantity = float(row[1])
            btc_avg_price = float(row[2])

            # Check if we have enough USDT
            if usdt_balance < amount_usdt:
                print(f"[Portfolio] Cannot buy: need ${amount_usdt:.2f} but have ${usdt_balance:.2f}")
                return None

            # Calculate new average price (weighted average)
            total_cost_before = btc_quantity * btc_avg_price
            total_cost_after = total_cost_before + amount_usdt
            new_quantity = btc_quantity + btc_to_buy
            new_avg_price = total_cost_after / new_quantity if new_quantity > 0 else 0

            # Update portfolio
            new_usdt = usdt_balance - amount_usdt
            cur.execute("""
                UPDATE portfolio
                SET usdt_balance = %s, btc_quantity = %s, btc_avg_price = %s, updated_at = NOW()
                WHERE model_id = %s
            """, (new_usdt, new_quantity, new_avg_price, MODEL_ID))

        conn.commit()
    finally:
        conn.close()

    trade = {
        "action": "BUY",
        "amount_usdt": round(amount_usdt, 2),
        "btc_quantity": round(btc_to_buy, 8),
        "price": current_price,
    }

    log_event("INFO",
              f"BUY {btc_to_buy:.6f} BTC @ ${current_price:,.2f} (spent ${amount_usdt:.2f})",
              model_id=MODEL_ID)
    print(f"[Portfolio] BUY {btc_to_buy:.6f} BTC @ ${current_price:,.2f} | "
          f"Remaining USDT: ${new_usdt:,.2f}")

    return trade


def sell_btc(btc_to_sell: float, current_price: float) -> dict:
    """
    Sell BTC for USDT.

    Args:
        btc_to_sell: how much BTC to sell (e.g., 0.002)
        current_price: current BTC price

    Returns:
        Trade record dict, or None if insufficient BTC.
    """
    _ensure_portfolio_exists()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Get current portfolio
            cur.execute("""
                SELECT usdt_balance, btc_quantity, btc_avg_price
                FROM portfolio WHERE model_id = %s FOR UPDATE
            """, (MODEL_ID,))
            row = cur.fetchone()
            usdt_balance = float(row[0])
            btc_quantity = float(row[1])
            btc_avg_price = float(row[2])

            # Check if we have enough BTC
            if btc_quantity < btc_to_sell:
                print(f"[Portfolio] Cannot sell: want {btc_to_sell:.6f} BTC but hold {btc_quantity:.6f}")
                return None

            # Calculate USDT received and P&L
            usdt_received = btc_to_sell * current_price
            cost_basis = btc_to_sell * btc_avg_price
            pnl = usdt_received - cost_basis

            # Update portfolio
            new_btc = btc_quantity - btc_to_sell
            new_usdt = usdt_balance + usdt_received
            # Average price stays the same (only changes on buys)

            cur.execute("""
                UPDATE portfolio
                SET usdt_balance = %s, btc_quantity = %s, updated_at = NOW()
                WHERE model_id = %s
            """, (new_usdt, new_btc, MODEL_ID))

        conn.commit()
    finally:
        conn.close()

    trade = {
        "action": "SELL",
        "amount_usdt": round(usdt_received, 2),
        "btc_quantity": round(btc_to_sell, 8),
        "price": current_price,
        "pnl": round(pnl, 2),
    }

    emoji = "✅" if pnl >= 0 else "❌"
    log_event("INFO",
              f"SELL {btc_to_sell:.6f} BTC @ ${current_price:,.2f} | "
              f"P&L: ${pnl:+,.2f}",
              model_id=MODEL_ID)
    print(f"[Portfolio] {emoji} SELL {btc_to_sell:.6f} BTC @ ${current_price:,.2f} | "
          f"P&L: ${pnl:+,.2f} | USDT: ${new_usdt:,.2f}")

    return trade


def sell_all_btc(current_price: float) -> dict:
    """Sell entire BTC holding."""
    portfolio = get_portfolio()
    if portfolio["btc_quantity"] <= 0:
        print("[Portfolio] No BTC to sell.")
        return None
    return sell_btc(portfolio["btc_quantity"], current_price)


def record_trade(trade: dict, prediction: dict):
    """Save a trade record to the trades table for history tracking."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                  (model_id, action, amount_usdt, btc_quantity, price,
                   predicted_high, predicted_low, direction_signal, confidence, pnl)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                MODEL_ID,
                trade["action"],
                trade["amount_usdt"],
                trade["btc_quantity"],
                trade["price"],
                prediction.get("predicted_high"),
                prediction.get("predicted_low"),
                prediction.get("direction"),
                prediction.get("confidence"),
                trade.get("pnl", 0),
            ))
        conn.commit()
    finally:
        conn.close()

def record_hold(prediction: dict, reason: str):
    """Save a HOLD record to the trades table for history tracking (UI visibility)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                  (model_id, action, amount_usdt, btc_quantity, price,
                   predicted_high, predicted_low, direction_signal, confidence, pnl)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                MODEL_ID,
                "HOLD",
                0,
                0,
                prediction["current_price"],
                prediction.get("predicted_high"),
                prediction.get("predicted_low"),
                prediction.get("direction"),
                prediction.get("confidence"),
                0,
            ))
        conn.commit()
    finally:
        conn.close()
