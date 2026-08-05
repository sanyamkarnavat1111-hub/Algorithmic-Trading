"""
scheduler/job_runner.py — 15-minute heartbeat scheduler.

Every 15 minutes:
  1. Fetch latest 15-min candle from Binance
  2. Run both models (Direction + Range)
  3. Decision engine decides: BUY, SELL, or HOLD
  4. Log prediction for accuracy tracking
  5. Check if retraining is due
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from data.binance_fetcher import fetch_latest
from data.database import log_event, get_connection
from config import HEARTBEAT_MINUTES

TIMEFRAME = "15m"
MODEL_ID = "ai_15m"


def heartbeat():
    """Main 15-minute job."""
    print(f"\n[Heartbeat] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    try:
        # Step 1: Fetch latest candle
        fetch_latest(TIMEFRAME)

        # Step 2: Run prediction
        from models.predictor import predict
        prediction = predict(TIMEFRAME)

        if prediction is None:
            print("[Heartbeat] No prediction (models not ready).", flush=True)
            return

        print(f"[Heartbeat] {prediction['direction']} "
              f"(conf={prediction['confidence']:.1%}) | "
              f"Price=${prediction['current_price']:,.0f} | "
              f"High=${prediction['predicted_high']:,.0f} | "
              f"Low=${prediction['predicted_low']:,.0f}", flush=True)

        # Step 3: Decision engine
        from decision.engine import make_decision
        decision = make_decision(prediction)
        print(f"[Heartbeat] Action: {decision['action']} — {decision['reason']}", flush=True)

        # Step 4: Log prediction for accuracy tracking
        _log_prediction(prediction)

        # Step 5: Check retraining
        _check_retrain()

    except Exception as e:
        try:
            log_event("ERROR", f"Heartbeat error: {str(e)}", model_id=MODEL_ID)
        except Exception:
            pass  # Don't crash if logging fails
        print(f"[Heartbeat] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print(f"[Heartbeat] Done.\n", flush=True)


def _log_prediction(prediction: dict):
    """Save prediction to predictions_log for accuracy tracking later."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions_log
                  (model_id, current_price, direction, confidence, predicted_high, predicted_low)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                MODEL_ID,
                prediction["current_price"],
                prediction["direction"],
                prediction["confidence"],
                prediction["predicted_high"],
                prediction["predicted_low"],
            ))
        conn.commit()
    except Exception as e:
        print(f"[Heartbeat] Failed to log prediction: {e}", flush=True)
    finally:
        conn.close()


def _check_retrain():
    """Check if retraining is due (every 100 closed trades)."""
    try:
        from learning.retrain_loop import check_and_retrain
        check_and_retrain()
    except Exception as e:
        print(f"[Heartbeat] Retrain check error: {e}", flush=True)


def start_scheduler():
    """Start the background scheduler."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        heartbeat,
        trigger=IntervalTrigger(minutes=HEARTBEAT_MINUTES),
        id="heartbeat_15m",
        replace_existing=True,
        max_instances=1,
        next_run_time=None,  # Don't run immediately
    )
    scheduler.start()
    print(f"[Scheduler] Started — every {HEARTBEAT_MINUTES} minutes.", flush=True)
    return scheduler
