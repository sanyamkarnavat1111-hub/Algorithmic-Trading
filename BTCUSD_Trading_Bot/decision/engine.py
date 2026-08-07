"""
decision/engine.py — Trading logic.

Rules:
  - Only BUY positions (no shorting)
  - Direction=BUY → open new position ($100-200)
  - Direction=SELL or HOLD → don't open, but check exits for open positions
  - Take profit: +10% above entry → sell
  - Stop loss: -5% below entry → sell (actual price OR AI predicts it will happen)
  - Max 5 positions open, 1 new per candle (staggered)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from data.database import get_connection, log_event
from config import TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX

MODEL_ID = "ai_15m"
MAX_POSITIONS = 5
TAKE_PROFIT_PCT = 0.10   # +10%
STOP_LOSS_PCT = -0.05    # -5%


def get_open_positions() -> list:
    """Get all currently open positions."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, entry_price, amount_usdt, btc_quantity, opened_at
                FROM positions
                WHERE model_id = %s AND status = 'OPEN'
                ORDER BY opened_at ASC
            """, (MODEL_ID,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r[0],
            "entry_price": float(r[1]),
            "amount_usdt": float(r[2]),
            "btc_quantity": float(r[3]),
            "opened_at": r[4],
        }
        for r in rows
    ]


def open_position(current_price: float, prediction: dict) -> dict:
    """Open a new BUY position."""
    amount_usdt = round(random.uniform(TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX), 2)
    btc_quantity = round(amount_usdt / current_price, 8)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO positions
                  (model_id, direction, entry_price, amount_usdt, btc_quantity,
                   predicted_high, predicted_low, confidence, status)
                VALUES (%s, 'BUY', %s, %s, %s, %s, %s, %s, 'OPEN')
                RETURNING id
            """, (MODEL_ID, current_price, amount_usdt, btc_quantity,
                  prediction["predicted_high"], prediction["predicted_low"],
                  prediction["confidence"]))
            pos_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    log_event("INFO",
              f"OPENED BUY #{pos_id} @ ${current_price:,.2f} | ${amount_usdt:.0f}",
              model_id=MODEL_ID)

    return {"id": pos_id, "entry_price": current_price, "amount_usdt": amount_usdt}


def close_position(position: dict, current_price: float, reason: str):
    """Close a position and record P&L."""
    entry_price = position["entry_price"]
    btc_quantity = position["btc_quantity"]
    pnl = round((current_price - entry_price) * btc_quantity, 4)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE positions
                SET exit_price = %s, pnl = %s, status = 'CLOSED',
                    close_reason = %s, closed_at = NOW()
                WHERE id = %s
            """, (current_price, pnl, reason, position["id"]))
        conn.commit()
    finally:
        conn.close()

    emoji = "✅" if pnl >= 0 else "❌"
    log_event("INFO",
              f"{emoji} CLOSED #{position['id']} @ ${current_price:,.2f} | "
              f"P&L: ${pnl:+,.2f} | {reason}",
              model_id=MODEL_ID)

    return pnl


def make_decision(prediction: dict) -> list:
    """
    Main decision logic. Called every 15 minutes.

    Returns list of actions taken (for logging/display).
    """
    if prediction is None:
        return [{"action": "SKIP", "reason": "No prediction available"}]

    direction = prediction["direction"]
    confidence = prediction["confidence"]
    current_price = prediction["current_price"]
    predicted_low = prediction["predicted_low"]

    actions = []
    open_positions = get_open_positions()

    # ── Step 1: Check all open positions for exit conditions ──────────────────

    for pos in open_positions:
        entry_price = pos["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price

        # Take Profit: +10%
        if pnl_pct >= TAKE_PROFIT_PCT:
            pnl = close_position(pos, current_price, "TAKE_PROFIT")
            actions.append({
                "action": "CLOSED",
                "reason": f"🎯 +{pnl_pct*100:.1f}% profit! Sold #{pos['id']} | P&L: ${pnl:+,.2f}",
                "pnl": pnl,
            })
            continue

        # Stop Loss: actual -5%
        if pnl_pct <= STOP_LOSS_PCT:
            pnl = close_position(pos, current_price, "STOP_LOSS")
            actions.append({
                "action": "CLOSED",
                "reason": f"❌ {pnl_pct*100:.1f}% loss. Sold #{pos['id']} | P&L: ${pnl:+,.2f}",
                "pnl": pnl,
            })
            continue

        # AI-assisted early exit: direction=SELL AND predicted low would hit -5%
        if direction == "SELL":
            predicted_loss_pct = (predicted_low - entry_price) / entry_price
            if predicted_loss_pct <= STOP_LOSS_PCT:
                pnl = close_position(pos, current_price, "AI_EARLY_EXIT")
                actions.append({
                    "action": "CLOSED",
                    "reason": f"⚠️ AI predicts -5% coming for #{pos['id']}. Sold early | P&L: ${pnl:+,.2f}",
                    "pnl": pnl,
                })
                continue

        # Hold
        actions.append({
            "action": "HOLD",
            "reason": f"Holding #{pos['id']} ({pnl_pct*100:+.2f}%)",
            "pnl": None,
        })

    # ── Step 2: Open new position if conditions met ───────────────────────────

    # Recount after potential closures
    remaining_open = len([a for a in actions if a["action"] == "HOLD"])

    if remaining_open < MAX_POSITIONS and direction == "BUY":
        pos = open_position(current_price, prediction)
        actions.append({
            "action": "OPENED",
            "reason": f"📈 AI says BUY ({confidence:.1%}). Opened #{pos['id']} @ ${current_price:,.2f} | ${pos['amount_usdt']:.0f}",
            "pnl": None,
        })
    elif direction != "BUY":
        actions.append({
            "action": "NO_ENTRY",
            "reason": f"AI says {direction} ({confidence:.1%}). No new position.",
            "pnl": None,
        })

    return actions
