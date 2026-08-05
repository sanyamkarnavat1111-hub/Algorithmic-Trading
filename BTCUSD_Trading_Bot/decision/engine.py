"""
decision/engine.py — Position-based trading logic.

How it works:
  1. AI predicts direction + high/low for next 75 minutes
  2. If no open position and AI says BUY → OPEN a position (buy BTC)
  3. If no open position and AI says SELL → OPEN a short position (sell BTC)
  4. If there IS an open position → check if it should be CLOSED:
     - Price hit predicted high → close with profit (target reached)
     - Price hit predicted low → close with loss (prediction was wrong)
     - AI reversed direction → close at current price
     - 5 candles (75 min) passed → close at current price (time expired)
  5. P&L = exit_price - entry_price (for BUY positions)
         = entry_price - exit_price (for SELL positions)

One position at a time. Clear open → hold → close lifecycle.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import datetime, timezone, timedelta
from data.database import get_connection, log_event
from config import TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX, LABEL_LOOKAHEAD, HEARTBEAT_MINUTES


MODEL_ID = "ai_15m"

# Position expires after this many minutes (5 candles × 15 min = 75 min)
POSITION_EXPIRY_MINUTES = LABEL_LOOKAHEAD * HEARTBEAT_MINUTES


def get_open_position() -> dict:
    """Get the currently open position, or None if no position is open."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, direction, entry_price, amount_usdt, btc_quantity,
                       predicted_high, predicted_low, confidence, opened_at
                FROM positions
                WHERE model_id = %s AND status = 'OPEN'
                ORDER BY opened_at DESC LIMIT 1
            """, (MODEL_ID,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "direction": row[1],
        "entry_price": float(row[2]),
        "amount_usdt": float(row[3]),
        "btc_quantity": float(row[4]),
        "predicted_high": float(row[5]),
        "predicted_low": float(row[6]),
        "confidence": float(row[7]),
        "opened_at": row[8],
    }


def open_position(prediction: dict) -> dict:
    """
    Open a new position based on the AI's prediction.

    BUY position: we buy BTC expecting price to go up
    SELL position: we sell BTC expecting price to go down
    """
    direction = prediction["direction"]
    current_price = prediction["current_price"]
    predicted_high = prediction["predicted_high"]
    predicted_low = prediction["predicted_low"]
    confidence = prediction["confidence"]

    # Random amount between $100-200
    amount_usdt = round(random.uniform(TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX), 2)
    btc_quantity = round(amount_usdt / current_price, 8)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO positions
                  (model_id, direction, entry_price, amount_usdt, btc_quantity,
                   predicted_high, predicted_low, confidence, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'OPEN')
                RETURNING id, opened_at
            """, (MODEL_ID, direction, current_price, amount_usdt, btc_quantity,
                  predicted_high, predicted_low, confidence))
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()

    position = {
        "id": row[0],
        "direction": direction,
        "entry_price": current_price,
        "amount_usdt": amount_usdt,
        "btc_quantity": btc_quantity,
        "predicted_high": predicted_high,
        "predicted_low": predicted_low,
        "opened_at": row[1],
    }

    log_event("INFO",
              f"OPENED {direction} @ ${current_price:,.2f} | ${amount_usdt:.0f} | "
              f"Target: ${predicted_high:,.0f} | Floor: ${predicted_low:,.0f}",
              model_id=MODEL_ID)

    return position


def close_position(position: dict, exit_price: float, reason: str) -> dict:
    """
    Close an open position and calculate P&L.

    P&L for BUY:  (exit_price - entry_price) × btc_quantity
    P&L for SELL: (entry_price - exit_price) × btc_quantity
    """
    entry_price = position["entry_price"]
    btc_quantity = position["btc_quantity"]
    direction = position["direction"]

    if direction == "BUY":
        pnl = (exit_price - entry_price) * btc_quantity
    else:  # SELL
        pnl = (entry_price - exit_price) * btc_quantity

    pnl = round(pnl, 4)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE positions
                SET exit_price = %s, pnl = %s, status = 'CLOSED',
                    close_reason = %s, closed_at = NOW()
                WHERE id = %s
            """, (exit_price, pnl, reason, position["id"]))
        conn.commit()
    finally:
        conn.close()

    emoji = "✅" if pnl >= 0 else "❌"
    log_event("INFO",
              f"{emoji} CLOSED {direction} @ ${exit_price:,.2f} | "
              f"P&L: ${pnl:+,.2f} | Reason: {reason}",
              model_id=MODEL_ID)

    return {"pnl": pnl, "reason": reason, "exit_price": exit_price}


def make_decision(prediction: dict) -> dict:
    """
    Main decision logic. Called every 15 minutes.

    Returns:
        {
            "action": "OPENED_BUY" | "OPENED_SELL" | "CLOSED" | "HOLD",
            "reason": str,
            "pnl": float or None
        }
    """
    if prediction is None:
        return {"action": "HOLD", "reason": "No prediction available", "pnl": None}

    direction = prediction["direction"]
    confidence = prediction["confidence"]
    current_price = prediction["current_price"]
    predicted_high = prediction["predicted_high"]
    predicted_low = prediction["predicted_low"]

    # ── Check if we have an open position ─────────────────────────────────────
    open_pos = get_open_position()

    if open_pos:
        # We have an open position — check if we should close it

        entry_price = open_pos["entry_price"]
        pos_direction = open_pos["direction"]
        pos_predicted_high = open_pos["predicted_high"]
        pos_predicted_low = open_pos["predicted_low"]
        opened_at = open_pos["opened_at"]

        # Reason 1: Price hit predicted high (target reached)
        if current_price >= pos_predicted_high:
            result = close_position(open_pos, current_price, "TARGET_HIT")
            return {
                "action": "CLOSED",
                "reason": f"Price ${current_price:,.0f} hit target ${pos_predicted_high:,.0f}. P&L: ${result['pnl']:+,.2f}",
                "pnl": result["pnl"],
            }

        # Reason 2: Price hit predicted low (prediction was wrong)
        if current_price <= pos_predicted_low:
            result = close_position(open_pos, current_price, "FLOOR_HIT")
            return {
                "action": "CLOSED",
                "reason": f"Price ${current_price:,.0f} hit floor ${pos_predicted_low:,.0f}. P&L: ${result['pnl']:+,.2f}",
                "pnl": result["pnl"],
            }

        # Reason 3: AI reversed direction
        if (pos_direction == "BUY" and direction == "SELL") or \
           (pos_direction == "SELL" and direction == "BUY"):
            result = close_position(open_pos, current_price, "DIRECTION_REVERSED")
            return {
                "action": "CLOSED",
                "reason": f"AI reversed to {direction}. Closed {pos_direction} @ ${current_price:,.0f}. P&L: ${result['pnl']:+,.2f}",
                "pnl": result["pnl"],
            }

        # Reason 4: Position expired (75 minutes passed)
        if opened_at:
            # Make opened_at timezone-aware if it isn't
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            age_minutes = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60
            if age_minutes >= POSITION_EXPIRY_MINUTES:
                result = close_position(open_pos, current_price, "EXPIRED")
                return {
                    "action": "CLOSED",
                    "reason": f"Position expired after {int(age_minutes)} min. Closed @ ${current_price:,.0f}. P&L: ${result['pnl']:+,.2f}",
                    "pnl": result["pnl"],
                }

        # Position still open, not closing yet
        return {
            "action": "HOLD",
            "reason": f"Holding {pos_direction} position (entry ${entry_price:,.0f}, current ${current_price:,.0f})",
            "pnl": None,
        }

    # ── No open position — check if we should open one ────────────────────────

    if direction == "BUY":
        pos = open_position(prediction)
        return {
            "action": "OPENED_BUY",
            "reason": f"AI says BUY ({confidence:.1%}). Opened @ ${current_price:,.0f}. Target: ${predicted_high:,.0f}",
            "pnl": None,
        }

    if direction == "SELL":
        pos = open_position(prediction)
        return {
            "action": "OPENED_SELL",
            "reason": f"AI says SELL ({confidence:.1%}). Opened @ ${current_price:,.0f}. Floor: ${predicted_low:,.0f}",
            "pnl": None,
        }

    # HOLD — no action
    return {
        "action": "HOLD",
        "reason": f"AI says HOLD ({confidence:.1%}). No position opened.",
        "pnl": None,
    }
