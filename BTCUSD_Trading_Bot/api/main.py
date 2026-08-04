"""
api/main.py — FastAPI application entry point.

On startup (before the server accepts requests):
  1. Drop old tables and create fresh schema
  2. Fetch 15-min candle data from Binance (full history)
  3. Train both AI models (Direction + Range)
  4. Start the 15-min scheduler
  5. Server goes live

All steps are logged so you can watch progress in Render logs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
import traceback

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import API_HOST, API_PORT

# Path to the UI folder
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")


# ── Bootstrap: runs BEFORE server accepts traffic ─────────────────────────────

def bootstrap_system():
    """
    Full system setup — runs on every deploy.
    Drops old tables, creates fresh schema, fetches data, trains models.
    All output goes to Render logs.
    """
    print("\n" + "=" * 60, flush=True)
    print("[Bootstrap] Starting full system setup...", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Drop old tables, create fresh schema
    print("\n[Step 1/4] Creating fresh database tables...", flush=True)
    try:
        from data.database import get_connection, create_tables
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DROP TABLE IF EXISTS candles CASCADE;
                    DROP TABLE IF EXISTS features CASCADE;
                    DROP TABLE IF EXISTS trades CASCADE;
                    DROP TABLE IF EXISTS model_versions CASCADE;
                    DROP TABLE IF EXISTS model_store CASCADE;
                    DROP TABLE IF EXISTS scalers CASCADE;
                    DROP TABLE IF EXISTS app_logs CASCADE;
                    DROP TABLE IF EXISTS portfolio CASCADE;
                    DROP TABLE IF EXISTS predictions_log CASCADE;
                """)
            conn.commit()
            print("[Step 1/4] Old tables dropped.", flush=True)
        finally:
            conn.close()

        create_tables()
        print("[Step 1/4] ✅ Fresh tables created.", flush=True)
    except Exception as e:
        print(f"[Step 1/4] ❌ Database setup failed: {e}", flush=True)
        traceback.print_exc()
        return False

    # Step 2: Fetch 15-min candle data from Binance
    print("\n[Step 2/4] Fetching 15-min candle data from Binance...", flush=True)
    print("[Step 2/4] This will take several minutes (fetching from 2017)...", flush=True)
    try:
        from data.binance_fetcher import test_binance_connection, sync_timeframe

        if not test_binance_connection():
            print("[Step 2/4] ❌ Cannot reach Binance API. Aborting.", flush=True)
            return False

        sync_timeframe("15m")
        print("[Step 2/4] ✅ Data fetch complete.", flush=True)
    except Exception as e:
        print(f"[Step 2/4] ❌ Data fetch failed: {e}", flush=True)
        traceback.print_exc()
        return False

    # Step 3: Train Direction Model
    print("\n[Step 3/4] Training Direction Model (BUY/SELL/HOLD)...", flush=True)
    try:
        from models.direction_trainer import train as train_direction
        direction_result = train_direction(warm_start=False)
        print(f"[Step 3/4] ✅ Direction Model trained. F1={direction_result['f1_weighted']:.4f}", flush=True)
    except Exception as e:
        print(f"[Step 3/4] ❌ Direction training failed: {e}", flush=True)
        traceback.print_exc()
        return False

    # Step 4: Train Range Model (HIGH + LOW)
    print("\n[Step 4/4] Training Range Model (HIGH/LOW prediction)...", flush=True)
    try:
        from models.range_trainer import train as train_range
        range_result = train_range(warm_start=False)
        print(f"[Step 4/4] ✅ Range Model trained. "
              f"HIGH MAE=${range_result['mae_high']:.2f}, "
              f"LOW MAE=${range_result['mae_low']:.2f}", flush=True)
    except Exception as e:
        print(f"[Step 4/4] ❌ Range training failed: {e}", flush=True)
        traceback.print_exc()
        return False

    print("\n" + "=" * 60, flush=True)
    print("[Bootstrap] ✅ ALL DONE. System is ready.", flush=True)
    print("=" * 60 + "\n", flush=True)
    return True


# ── Startup & Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    print("\n[START] Crypto Trading Bot starting up...", flush=True)

    # Run bootstrap in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(None, bootstrap_system)

    if success:
        # Start scheduler only after bootstrap succeeds
        from scheduler.job_runner import start_scheduler
        scheduler = start_scheduler()
        print("[OK] Scheduler started (15-min heartbeat)", flush=True)
    else:
        scheduler = None
        print("[WARN] Bootstrap failed. Scheduler not started.", flush=True)

    yield  # App is running and accepting requests

    # Shutdown
    if scheduler:
        scheduler.shutdown(wait=False)
    print("[EXIT] Server stopped.", flush=True)


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="BTC Trading Bot",
    description="Self-learning BTC/USDT trading bot with dual AI models",
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static UI files
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health endpoint — keeps Render free tier alive."""
    return JSONResponse(content={
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "BTC Trading Bot v2",
    })


@app.get("/")
def serve_dashboard():
    """Serve the dashboard HTML page."""
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={"message": "Dashboard not found. Check ui/index.html"})


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
