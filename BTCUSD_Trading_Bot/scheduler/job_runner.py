"""
scheduler/job_runner.py — 15-minute heartbeat scheduler.

Every 15 minutes:
  1. Fetch latest 15-min candle from Binance
  2. Run both models (Direction + Range)
  3. Log prediction
  4. (Phase 2: Execute trade decisions)

For now (Phase 1), it just fetches data and logs predictions.
Trade execution will be added in Phase 2.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from data.binance_fetcher import fetch_latest
from data.database import log_event
from config import HEARTBEAT_MINUTES

TIMEFRAME = "15m"
MODEL_ID = "ai_15m"


def heartbeat():
    """Main 15-minute job."""
    print(f"\n[Scheduler] Heartbeat at {datetime.now().strftime('%H:%M:%S')}", flush=True)

    try:
        # Step 1: Fetch latest candle
        fetch_latest(TIMEFRAME)

        # Step 2: Run prediction (if models exist)
        try:
            from models.predictor import predict
            prediction = predict(TIMEFRAME)
            if prediction:
                print(f"[Scheduler] Prediction: {prediction['direction']} "
                      f"(conf={prediction['confidence']:.1%}) | "
                      f"High=${prediction['predicted_high']:,.0f} | "
                      f"Low=${prediction['predicted_low']:,.0f}", flush=True)
            else:
                print("[Scheduler] No prediction (models not ready)", flush=True)
        except Exception as e:
            print(f"[Scheduler] Prediction error: {e}", flush=True)

    except Exception as e:
        log_event("ERROR", f"Heartbeat error: {str(e)}", model_id=MODEL_ID)
        print(f"[Scheduler] Error: {e}", flush=True)

    print(f"[Scheduler] Heartbeat complete.", flush=True)


def start_scheduler():
    """Start the background scheduler. Called once after bootstrap."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        heartbeat,
        trigger=IntervalTrigger(minutes=HEARTBEAT_MINUTES),
        id="heartbeat_15m",
        replace_existing=True,
        max_instances=1,
        next_run_time=None,  # Don't run immediately (bootstrap just ran)
    )
    scheduler.start()
    print(f"[Scheduler] Started — running every {HEARTBEAT_MINUTES} minutes", flush=True)
    return scheduler
