"""
trading/paper_trader.py — Simulates trades with fake money.

No real orders are placed. All trades exist only in PostgreSQL.
This is the safe environment where AI learns before real money.

Flow:
  1. Predictor fires a BUY/SELL signal
  2. risk_manager checks all safety rules
  3. If clear → open a paper trade in DB
  4. On next heartbeat → check if stop-loss hit or exit signal
  5. Close trade → record P&L → feed back to retraining loop
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from data.database import get_connection, log_event
from trading.risk_manager import (
    calculate_position_size,
    calculate_stop_loss,
    has_open_trade,
    is_circuit_breaker_active,
    should_stop_loss_trigger,
)
from config import TIMEFRAMES


def open_trade(prediction: dict) -> dict | None:
    """
    Open a new paper trade based on a model prediction.

    Args:
        prediction: dict from predictor.predict() containing:
                    model_id, signal, confidence, current_price, atr

    Returns:
        The new trade record dict, or None if trade was blocked by risk rules.
    """
    model_id      = prediction["model_id"]
    signal        = prediction["signal"]
    confidence    = prediction["confidence"]
    entry_price   = prediction["current_price"]
    atr           = prediction["atr"]
    timeframe     = prediction["timeframe"]

    # ── Safety checks ─────────────────────────────────────────────────────────
    if not prediction.get("should_trade"):
        return None  # Confidence too low

    if has_open_trade(model_id):
        return None  # Already in a trade

    if is_circuit_breaker_active(model_id):
        return None  # Daily loss limit hit

    # ── Calculate position ────────────────────────────────────────────────────
    position_size = calculate_position_size(model_id, entry_price)
    stop_loss     = calculate_stop_loss(signal, entry_price, atr)

    # ── Save trade to PostgreSQL ──────────────────────────────────────────────
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                  (model_id, signal, confidence, entry_price, stop_loss, position_size, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'OPEN')
                RETURNING id, opened_at
            """, (model_id, signal, confidence, entry_price, stop_loss, position_size))
            row = cur.fetchone()
            trade_id   = row[0]
            opened_at  = row[1]
        conn.commit()
    finally:
        conn.close()

    trade = {
        "id":            trade_id,
        "model_id":      model_id,
        "signal":        signal,
        "confidence":    confidence,
        "entry_price":   entry_price,
        "stop_loss":     stop_loss,
        "position_size": position_size,
        "status":        "OPEN",
        "opened_at":     opened_at.isoformat(),
    }

    log_event("INFO",
              f"Opened {signal} @ {entry_price} | SL={stop_loss} | size={position_size:.6f} BTC",
              model_id=model_id)
    print(f"[PaperTrader] {model_id}: Opened {signal} @ {entry_price:.2f} | SL={stop_loss:.2f}")
    return trade


def record_skipped_trade(prediction: dict):
    """
    Record a HOLD or low-confidence prediction so it appears in the UI history,
    without affecting P&L or actual trading positions.
    """
    model_id      = prediction["model_id"]
    signal        = prediction["signal"]
    confidence    = prediction["confidence"]
    current_price = prediction["current_price"]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades
                  (model_id, signal, confidence, entry_price, exit_price, stop_loss, position_size, pnl, pnl_pct, status, opened_at, closed_at)
                VALUES (%s, %s, %s, %s, %s, 0, 0, 0, 0, 'SKIPPED', NOW(), NOW())
            """, (model_id, signal, confidence, current_price, current_price))
        conn.commit()
    finally:
        conn.close()


def check_and_close_trades(model_id: str, current_price: float, exit_signal: str = None):
    """
    Check all open trades for a model. Close them if:
      - Stop-loss has been hit
      - Model fires an opposite signal (e.g., BUY trade but model now says SELL)

    Args:
        model_id:     e.g., 'ai_1h'
        current_price: Latest BTC price
        exit_signal:  The current model signal (BUY/SELL/HOLD)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, signal, entry_price, stop_loss, position_size
                FROM trades
                WHERE model_id = %s AND status = 'OPEN'
            """, (model_id,))
            open_trades = cur.fetchall()
    finally:
        conn.close()

    for trade_row in open_trades:
        trade_id, signal, entry_price, stop_loss, position_size = trade_row
        trade = {
            "id":           trade_id,
            "signal":       signal,
            "entry_price":  float(entry_price),
            "stop_loss":    float(stop_loss),
            "position_size": float(position_size),
        }

        close_reason = None

        # Check stop-loss
        if should_stop_loss_trigger(trade, current_price):
            close_reason = "STOPPED_OUT"

        # Check exit signal (model reversed direction)
        elif exit_signal and exit_signal != "HOLD":
            if (signal == "BUY"  and exit_signal == "SELL") or \
               (signal == "SELL" and exit_signal == "BUY"):
                close_reason = "CLOSED"

        if close_reason:
            _close_trade(trade, current_price, close_reason, model_id)


def _close_trade(trade: dict, exit_price: float, status: str, model_id: str):
    """
    Close an open trade and calculate P&L.

    P&L for BUY:  (exit - entry) × size
    P&L for SELL: (entry - exit) × size  [short position]
    """
    entry_price   = trade["entry_price"]
    position_size = trade["position_size"]
    signal        = trade["signal"]

    if signal == "BUY":
        pnl = (exit_price - entry_price) * position_size
    else:  # SELL (short)
        pnl = (entry_price - exit_price) * position_size

    pnl_pct = pnl / (entry_price * position_size) if (entry_price * position_size) > 0 else 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades
                SET exit_price = %s,
                    pnl        = %s,
                    pnl_pct    = %s,
                    status     = %s,
                    closed_at  = NOW()
                WHERE id = %s
            """, (exit_price, round(pnl, 4), round(pnl_pct, 6), status, trade["id"]))
        conn.commit()
    finally:
        conn.close()

    emoji = "✅" if pnl >= 0 else "❌"
    log_event("INFO",
              f"Closed {signal} @ {exit_price:.2f} | P&L={pnl:.2f} USDT ({pnl_pct:.2%}) | {status}",
              model_id=model_id)
    print(f"[PaperTrader] {model_id}: {emoji} Closed {signal} @ {exit_price:.2f} | P&L={pnl:+.2f} USDT")


def get_open_trades(model_id: str) -> list:
    """Return all currently open trades for a model (used by dashboard)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, signal, confidence, entry_price, stop_loss, position_size, opened_at
                FROM trades
                WHERE model_id = %s AND status = 'OPEN'
                ORDER BY opened_at DESC
            """, (model_id,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id":            r[0],
            "signal":        r[1],
            "confidence":    float(r[2]),
            "entry_price":   float(r[3]),
            "stop_loss":     float(r[4]),
            "position_size": float(r[5]),
            "opened_at":     r[6].isoformat(),
        }
        for r in rows
    ]


def get_recent_trades(model_id: str, limit: int = 20) -> list:
    """Return recent closed trades for a model (used by dashboard)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, signal, confidence, entry_price, exit_price, pnl, pnl_pct, status, opened_at, closed_at
                FROM trades
                WHERE model_id = %s AND status != 'OPEN'
                ORDER BY closed_at DESC
                LIMIT %s
            """, (model_id, limit))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id":           r[0],
            "signal":       r[1],
            "confidence":   float(r[2]),
            "entry_price":  float(r[3]),
            "exit_price":   float(r[4]) if r[4] else None,
            "pnl":          float(r[5]) if r[5] else None,
            "pnl_pct":      float(r[6]) if r[6] else None,
            "status":       r[7],
            "opened_at":    r[8].isoformat(),
            "closed_at":    r[9].isoformat() if r[9] else None,
        }
        for r in rows
    ]


def get_total_pnl(model_id: str) -> dict:
    """Return total P&L stats for a model (used by dashboard)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                      AS total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)    AS wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)   AS losses,
                    COALESCE(SUM(pnl), 0)                        AS total_pnl,
                    COALESCE(AVG(pnl_pct) * 100, 0)             AS avg_pnl_pct
                FROM trades
                WHERE model_id = %s AND status IN ('CLOSED', 'STOPPED_OUT')
            """, (model_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    total = row[0] or 0
    wins  = row[1] or 0
    return {
        "total_trades": total,
        "wins":         wins,
        "losses":       row[2] or 0,
        "win_rate":     round(wins / total, 4) if total > 0 else 0,
        "total_pnl":    round(float(row[3]), 2),
        "avg_pnl_pct":  round(float(row[4]), 4),
    }
