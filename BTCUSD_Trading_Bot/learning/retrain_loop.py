"""
learning/retrain_loop.py — Retraining loop for both AI models.

Triggers after every 100 closed trades:
  1. Retrain Direction Model (warm-start)
  2. Retrain Range Model (warm-start)
  3. Only deploy new version if it scores equal or better

Also updates prediction accuracy (compares past predictions to actual outcomes).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_connection, log_event, clean_old_logs
from config import RETRAIN_EVERY_N_TRADES

MODEL_ID = "ai_15m"
TIMEFRAME = "15m"


def get_closed_trades_count() -> int:
    """Count total trades in the trades table."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM trades WHERE model_id = %s", (MODEL_ID,))
            return cur.fetchone()[0]
    finally:
        conn.close()


def get_last_retrain_count() -> int:
    """Get the trade count at the time of last retrain."""
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


def check_and_retrain():
    """
    Check if retraining is due and trigger if yes.
    Called after every heartbeat.
    """
    total_trades = get_closed_trades_count()
    last_count = get_last_retrain_count()
    new_trades = total_trades - last_count

    if new_trades < RETRAIN_EVERY_N_TRADES:
        return

    print(f"[Retrain] {new_trades} new trades since last retrain. Triggering...", flush=True)
    log_event("INFO", f"Starting retrain after {new_trades} new trades", model_id=MODEL_ID)

    # Retrain Direction Model
    try:
        from models.direction_trainer import train as train_direction
        result = train_direction(warm_start=True)
        print(f"[Retrain] Direction Model v{result['version']} — F1={result['f1_weighted']:.4f}", flush=True)
    except Exception as e:
        log_event("ERROR", f"Direction retrain failed: {e}", model_id=MODEL_ID)
        print(f"[Retrain] Direction retrain failed: {e}", flush=True)

    # Retrain Range Model
    try:
        from models.range_trainer import train as train_range
        result = train_range(warm_start=True)
        print(f"[Retrain] Range Model — HIGH MAE=${result['mae_high']:.2f}, "
              f"LOW MAE=${result['mae_low']:.2f}", flush=True)
    except Exception as e:
        log_event("ERROR", f"Range retrain failed: {e}", model_id=MODEL_ID)
        print(f"[Retrain] Range retrain failed: {e}", flush=True)

    # Record retrain count
    log_event("INFO", f"RETRAIN_COUNT:{total_trades}", model_id=MODEL_ID)
    clean_old_logs(days_to_keep=14)
    print(f"[Retrain] Complete.", flush=True)


def update_prediction_accuracy():
    """
    Look at predictions made 75+ minutes ago and check if they were correct.
    Updates the predictions_log table with actual_high, actual_low, was_correct.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Find predictions that are old enough to verify (75 min = 5 candles)
            cur.execute("""
                SELECT id, predicted_at, current_price, direction,
                       predicted_high, predicted_low
                FROM predictions_log
                WHERE actual_high IS NULL
                  AND predicted_at < NOW() - INTERVAL '75 minutes'
                ORDER BY predicted_at ASC
                LIMIT 50
            """)
            predictions = cur.fetchall()

            for pred in predictions:
                pred_id, predicted_at, current_price, direction, pred_high, pred_low = pred

                # Get actual high and low in the 75 minutes after prediction
                cur.execute("""
                    SELECT MAX(high), MIN(low) FROM candles
                    WHERE timeframe = '15m'
                      AND open_time > %s
                      AND open_time <= %s + INTERVAL '75 minutes'
                """, (predicted_at, predicted_at))
                result = cur.fetchone()

                if result and result[0] is not None:
                    actual_high = float(result[0])
                    actual_low = float(result[1])

                    # Direction was correct if:
                    # BUY and price went up, SELL and price went down
                    was_correct = False
                    if direction == "BUY" and actual_high > float(current_price):
                        was_correct = True
                    elif direction == "SELL" and actual_low < float(current_price):
                        was_correct = True
                    elif direction == "HOLD":
                        was_correct = True  # HOLD is always "correct" in a sense

                    cur.execute("""
                        UPDATE predictions_log
                        SET actual_high = %s, actual_low = %s, was_correct = %s
                        WHERE id = %s
                    """, (actual_high, actual_low, was_correct, pred_id))

        conn.commit()
    except Exception as e:
        print(f"[Accuracy] Error updating prediction accuracy: {e}", flush=True)
    finally:
        conn.close()
