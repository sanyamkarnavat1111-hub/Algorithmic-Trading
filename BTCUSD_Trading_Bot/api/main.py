"""
api/main.py — FastAPI application entry point.

Starts up:
  - Database tables (create if not exist)
  - Background scheduler (15-min heartbeat)
  - FastAPI with dashboard routes
  - Serves the UI from /ui folder
  - /health endpoint for Render cron/UptimeRobot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from data.database import create_tables
from scheduler.job_runner import start_scheduler
from api.routes import router
from config import API_HOST, API_PORT

# Path to the UI folder (relative to this file's directory)
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")


# ── Startup & Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    print("\n[START] Crypto Trading Bot starting up...")

    # Create DB tables
    create_tables()
    print("[OK] Database tables ready")

    # Test Binance API connection immediately
    from data.binance_fetcher import test_binance_connection
    test_binance_connection()

    # Start background scheduler
    scheduler = start_scheduler()
    print("[OK] Scheduler started")

    # Auto-bootstrap if first run
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, bootstrap_system)

    yield  # App is running

    # Shutdown
    scheduler.shutdown(wait=False)
    print("[EXIT] Scheduler stopped. Bye!")


# ── App ───────────────────────────────────────────────────────────────────────

def bootstrap_system():
    """Check if model exists, if not, fetch data and train initial model."""
    try:
        from models.registry import get_model_info
        from data.binance_fetcher import sync_timeframe
        from models.trainer import train

        info = get_model_info("1h")
        if info["version"] is None:
            print("\n[Bootstrap] ⚠️ First run detected! DB is empty.", flush=True)
            print("[Bootstrap] Fetching historical data (this will take a minute)...", flush=True)
            try:
                sync_timeframe("1h")
                print("[Bootstrap] Data fetched. Training initial AI model...", flush=True)
                train("1h")
                print("[Bootstrap] ✅ Initial setup complete! Bot is now fully autonomous.\n", flush=True)
            except Exception as e:
                print(f"[Bootstrap] Error during fetch/train: {e}", flush=True)
        else:
            print(f"[Bootstrap] Found existing AI model (v{info['version']}). Resuming operations.", flush=True)
    except Exception as e:
        print(f"[Bootstrap] CRITICAL THREAD ERROR: {e}", flush=True)


app = FastAPI(
    title="Crypto Trading Bot",
    description="Self-learning algorithmic trading bot — 3 parallel AI models on BTC/USDT",
    version="1.0.0",
    lifespan=lifespan,
)

# Register API routes
app.include_router(router)

# Serve static UI files (JS, CSS)
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


# ── Core Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health endpoint — keep Render free tier alive.
    Ping this every 14 minutes via UptimeRobot or Render cron job.

    Returns: {"status": "ok", "timestamp": "..."}
    """
    return JSONResponse(content={
        "status":    "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service":   "Crypto Trading Bot",
    })


@app.get("/")
def serve_dashboard():
    """Serve the main dashboard HTML page."""
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={"message": "Dashboard UI not found. Check ui/index.html"})


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
