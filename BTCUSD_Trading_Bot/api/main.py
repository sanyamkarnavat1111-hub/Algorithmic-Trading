"""
api/main.py — FastAPI application entry point.

Startup order (fixes Render port scan timeout):
  1. Server starts IMMEDIATELY and binds to port (Render sees it's alive)
  2. Background thread runs: drop tables → fetch data → train models
  3. Once background setup finishes, scheduler starts

This way Render doesn't timeout waiting for port binding.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
import threading
import traceback

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import API_HOST, API_PORT

# Path to the UI folder
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")

# Global state to track bootstrap progress
bootstrap_status = {"state": "pending", "message": "Waiting to start..."}


# ── Bootstrap: runs in BACKGROUND THREAD after server is already live ─────────

def bootstrap_system():
    """
    Full system setup — runs in background after server binds to port.
    Drops old tables, creates fresh schema, fetches data, trains models.
    All output goes to Render logs.
    """
    global bootstrap_status

    bootstrap_status = {"state": "running", "message": "Starting setup..."}
    print("\n" + "=" * 60, flush=True)
    print("[Bootstrap] Starting full system setup (background thread)...", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Drop old tables, create fresh schema
    bootstrap_status["message"] = "Step 1/4: Creating database tables..."
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
        bootstrap_status = {"state": "failed", "message": f"DB setup failed: {e}"}
        print(f"[Step 1/4] ❌ Database setup failed: {e}", flush=True)
        traceback.print_exc()
        return

    # Step 2: Fetch 15-min candle data from Binance
    bootstrap_status["message"] = "Step 2/4: Fetching candle data from Binance..."
    print("\n[Step 2/4] Fetching 15-min candle data from Binance...", flush=True)
    print("[Step 2/4] This will take several minutes (fetching from 2017)...", flush=True)
    try:
        from data.binance_fetcher import test_binance_connection, sync_timeframe

        if not test_binance_connection():
            bootstrap_status = {"state": "failed", "message": "Cannot reach Binance API"}
            print("[Step 2/4] ❌ Cannot reach Binance API.", flush=True)
            return

        sync_timeframe("15m")
        print("[Step 2/4] ✅ Data fetch complete.", flush=True)
    except Exception as e:
        bootstrap_status = {"state": "failed", "message": f"Data fetch failed: {e}"}
        print(f"[Step 2/4] ❌ Data fetch failed: {e}", flush=True)
        traceback.print_exc()
        return

    # Step 3: Train Direction Model
    bootstrap_status["message"] = "Step 3/4: Training Direction Model..."
    print("\n[Step 3/4] Training Direction Model (BUY/SELL/HOLD)...", flush=True)
    try:
        from models.direction_trainer import train as train_direction
        direction_result = train_direction(warm_start=False)
        print(f"[Step 3/4] ✅ Direction Model trained. "
              f"F1={direction_result['f1_weighted']:.4f}", flush=True)
    except Exception as e:
        bootstrap_status = {"state": "failed", "message": f"Direction training failed: {e}"}
        print(f"[Step 3/4] ❌ Direction training failed: {e}", flush=True)
        traceback.print_exc()
        return

    # Step 4: Train Range Model (HIGH + LOW)
    bootstrap_status["message"] = "Step 4/4: Training Range Model..."
    print("\n[Step 4/4] Training Range Model (HIGH/LOW prediction)...", flush=True)
    try:
        from models.range_trainer import train as train_range
        range_result = train_range(warm_start=False)
        print(f"[Step 4/4] ✅ Range Model trained. "
              f"HIGH MAE=${range_result['mae_high']:.2f}, "
              f"LOW MAE=${range_result['mae_low']:.2f}", flush=True)
    except Exception as e:
        bootstrap_status = {"state": "failed", "message": f"Range training failed: {e}"}
        print(f"[Step 4/4] ❌ Range training failed: {e}", flush=True)
        traceback.print_exc()
        return

    # Step 5: Start scheduler
    bootstrap_status["message"] = "Starting scheduler..."
    print("\n[Bootstrap] Starting 15-min scheduler...", flush=True)
    try:
        from scheduler.job_runner import start_scheduler
        start_scheduler()
        print("[Bootstrap] ✅ Scheduler started.", flush=True)
    except Exception as e:
        print(f"[Bootstrap] ⚠️ Scheduler failed to start: {e}", flush=True)

    bootstrap_status = {"state": "complete", "message": "System fully operational."}
    print("\n" + "=" * 60, flush=True)
    print("[Bootstrap] ✅ ALL DONE. Bot is fully operational.", flush=True)
    print("=" * 60 + "\n", flush=True)


# ── Startup & Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server starts FIRST (binds port immediately).
    Then bootstrap runs in background thread.
    """
    print("\n[START] Server starting — binding port immediately...", flush=True)

    # Start bootstrap in a background thread (non-blocking)
    thread = threading.Thread(target=bootstrap_system, daemon=True)
    thread.start()
    print("[START] Bootstrap started in background thread.", flush=True)
    print("[START] Server is LIVE. Check /health or /bootstrap-status.", flush=True)

    yield  # Server is running and accepting requests

    print("[EXIT] Server shutting down.", flush=True)


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="BTC Trading Bot",
    description="Self-learning BTC/USDT trading bot with dual AI models",
    version="2.0.0",
    lifespan=lifespan,
)

# Import and register API routes
from api.routes import router
app.include_router(router)

# Serve static UI files
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


# ── Core Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health endpoint — keeps Render free tier alive."""
    return JSONResponse(content={
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bootstrap": bootstrap_status["state"],
    })


@app.get("/bootstrap-status")
def get_bootstrap_status():
    """Check how far along the bootstrap process is."""
    return JSONResponse(content=bootstrap_status)


@app.get("/")
def serve_dashboard():
    """Serve the dashboard HTML page."""
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={
        "message": "Bot is starting up. Check /bootstrap-status for progress.",
        "bootstrap": bootstrap_status,
    })


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
