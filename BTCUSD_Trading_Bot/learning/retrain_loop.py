"""
learning/retrain_loop.py — Self-improving online learning loop for the 1H model.

Triggered after every RETRAIN_EVERY_N_TRADES closed trades.
Retrains the model with warm-start so it builds on what it already learned.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_connection, log_event, clean_old_logs
from models.trainer import train
from config import RETRAIN_EVERY_N_TRADES

TIMEFRAME = "1h"
MODEL_ID  = "ai_1h"


def get_closed_trades_count() -> int:
    """Count all closed trades for the 1H model."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM trades
                WHERE model_id = %s AND status != 'OPEN'
            """, (MODEL_ID,))
            return cur.fetchone()[0]
    finally:
        conn.close()


def get_last_retrain_count() -> int:
    """Read the trade count recorded at the time of last retrain."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT message FROM app_logs
                WHERE model_id = %s AND message LIKE 'RETRAIN_COUNT:%%'
                ORDER BY created_at DESC LIMIT 1
            """, (MODEL_ID,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return 0
    try:
        return int(row[0].split(":")[1])
    except (IndexError, ValueError):
        return 0


def check_and_retrain(timeframe: str = TIMEFRAME):
    """
    Check if the 1H model needs retraining and trigger warm-start if yes.
    Called after every heartbeat.
    """
    total_trades = get_closed_trades_count()
    last_count   = get_last_retrain_count()
    new_trades   = total_trades - last_count

    print(f"[Retrain] {MODEL_ID}: {new_trades} new trades since last retrain "
          f"(threshold={RETRAIN_EVERY_N_TRADES})")

    if new_trades < RETRAIN_EVERY_N_TRADES:
        return

    print(f"[Retrain] {MODEL_ID}: Triggering warm-start retraining...")
    log_event("INFO", f"Starting warm-start retrain after {new_trades} new trades",
              model_id=MODEL_ID)

    try:
        result = train(TIMEFRAME, warm_start=True)
        log_event("INFO", f"RETRAIN_COUNT:{total_trades}", model_id=MODEL_ID)
        log_event("INFO",
                  f"Retrain complete: v{result['version']} f1={result['f1_weighted']:.4f}",
                  model_id=MODEL_ID)
        print(f"[Retrain] {MODEL_ID}: Retrain complete - "
              f"v{result['version']} | f1={result['f1_weighted']:.4f}")
        clean_old_logs(days_to_keep=7)
    except Exception as e:
        log_event("ERROR", f"Retrain failed: {str(e)}", model_id=MODEL_ID)
        print(f"[Retrain] {MODEL_ID}: Retrain failed - {e}")
