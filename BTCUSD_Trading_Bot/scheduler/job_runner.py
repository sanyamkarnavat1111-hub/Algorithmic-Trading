"""
scheduler/job_runner.py — Heartbeat scheduler for the 1H AI model.

Runs every 15 minutes and performs:
  1. Fetch latest 1H candles from Binance
  2. Run prediction (BUY/SELL/HOLD)
  3. Check open trades for stop-loss / exit signals
  4. Open new trade if signal is confident enough
  5. Check if model needs retraining
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from data.binance_fetcher import fetch_latest
from models.predictor import predict
from trading.paper_trader import open_trade, check_and_close_trades
from learning.retrain_loop import check_and_retrain
from data.database import log_event
from config import HEARTBEAT_MINUTES

MODEL_ID  = "ai_1h"
TIMEFRAME = "1h"


def heartbeat():
    """Main 15-minute job for the 1H model."""
    print(f"\n[Scheduler] Heartbeat started")

    try:
        # Step 1 — Pull latest 1H candles from Binance
        fetch_latest(TIMEFRAME)

        # Step 2 — Get prediction from active model
        prediction = predict(TIMEFRAME)
        if prediction is None:
            print(f"[Scheduler] {MODEL_ID}: No prediction (model not trained yet)")
            return

        current_price = prediction["current_price"]
        signal        = prediction["signal"]
        confidence    = prediction["confidence"]
        print(f"[Scheduler] {MODEL_ID}: {signal} @ ${current_price:,.2f} (conf={confidence:.1%})")

        # Step 3 — Check and close existing trades
        check_and_close_trades(MODEL_ID, current_price, exit_signal=signal)

        # Step 4 — Open a new trade if signal is strong enough
        if prediction["should_trade"]:
            new_trade = open_trade(prediction)
            if new_trade:
                print(f"[Scheduler] {MODEL_ID}: Opened {signal} trade")

    except Exception as e:
        log_event("ERROR", f"Heartbeat error: {str(e)}", model_id=MODEL_ID)
        print(f"[Scheduler] {MODEL_ID}: Error - {e}")

    # Step 5 — Check if retraining is due
    try:
        check_and_retrain(TIMEFRAME)
    except Exception as e:
        log_event("ERROR", f"Retrain check error: {str(e)}", model_id=MODEL_ID)
        print(f"[Scheduler] Retrain check error - {e}")

    print(f"[Scheduler] Heartbeat complete\n")


def start_scheduler():
    """Start the background scheduler. Called once when FastAPI starts."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        heartbeat,
        trigger=IntervalTrigger(minutes=HEARTBEAT_MINUTES),
        id="heartbeat",
        replace_existing=True,
        max_instances=1,    # never run two heartbeats simultaneously
    )
    scheduler.start()
    print(f"[Scheduler] Started — running every {HEARTBEAT_MINUTES} minutes")
    return scheduler
