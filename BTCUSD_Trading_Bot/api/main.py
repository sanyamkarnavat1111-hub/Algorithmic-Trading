"""
api/main.py — FastAPI application entry point.

Startup order:
  1. Server starts IMMEDIATELY and binds to port (Render sees it's alive)
  2. Self-ping thread keeps server awake every 5 minutes
  3. User clicks buttons on dashboard to trigger: fetch data, train models
  4. No auto-drop of tables — incremental by design

The data fetch and training are triggered manually via the dashboard
so they can be resumed if the server ever spins down.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import threading
import time
import traceback

import requests as http_requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from config import API_HOST, API_PORT

# Path to the UI folder
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")

# ── Global State ──────────────────────────────────────────────────────────────

status = {
    "state": "idle",          # idle, fetching, training_direction, training_range, complete, error
    "message": "Server is up. Use dashboard to start.",
    "candles_fetched": 0,
    "direction_f1": None,
    "range_mae_high": None,
    "range_mae_low": None,
    "last_error": None,
}

# Lock to prevent running multiple operations at once
operation_lock = threading.Lock()


# ── Self-Ping: keeps Render free tier alive ───────────────────────────────────

def self_ping_loop():
    """
    Pings own /health endpoint every 5 minutes to prevent Render spin-down.
    More reliable than external cron jobs because it runs inside the process.
    """
    # Wait 30 seconds for server to fully start
    time.sleep(30)

    port = API_PORT
    url = f"http://localhost:{port}/health"

    while True:
        try:
            http_requests.get(url, timeout=5)
        except Exception:
            pass  # Server might not be ready yet, that's fine
        time.sleep(300)  # Every 5 minutes


# ── Operations (triggered by dashboard buttons) ──────────────────────────────

def run_setup_tables():
    """Create tables if they don't exist (does NOT drop existing data)."""
    global status
    try:
        from data.database import create_tables
        create_tables()
        status["message"] = "Tables ready."
        print("[Setup] ✅ Tables created/verified.", flush=True)
        return True
    except Exception as e:
        status["state"] = "error"
        status["last_error"] = str(e)
        print(f"[Setup] ❌ Table creation failed: {e}", flush=True)
        return False


def run_fetch_data():
    """Fetch 15-min candles from Binance. Resumes from last candle in DB."""
    global status

    if not operation_lock.acquire(blocking=False):
        status["message"] = "Another operation is already running."
        return

    try:
        status["state"] = "fetching"
        status["message"] = "Fetching 15-min candles from Binance (incremental)..."
        print("\n[Fetch] Starting 15-min candle fetch...", flush=True)

        run_setup_tables()

        from data.binance_fetcher import test_binance_connection, sync_timeframe

        if not test_binance_connection():
            status["state"] = "error"
            status["message"] = "Cannot reach Binance API."
            status["last_error"] = "Binance API unreachable"
            return

        sync_timeframe("15m")

        # Count candles in DB
        from data.database import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM candles WHERE timeframe = '15m'")
                count = cur.fetchone()[0]
        finally:
            conn.close()

        status["candles_fetched"] = count
        status["state"] = "idle"
        status["message"] = f"Fetch complete. {count:,} candles in DB."
        print(f"[Fetch] ✅ Done. {count:,} candles in database.", flush=True)

    except Exception as e:
        status["state"] = "error"
        status["message"] = f"Fetch failed: {str(e)}"
        status["last_error"] = str(e)
        print(f"[Fetch] ❌ Failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        operation_lock.release()


def run_train_direction():
    """Train the Direction Model (BUY/SELL/HOLD)."""
    global status

    if not operation_lock.acquire(blocking=False):
        status["message"] = "Another operation is already running."
        return

    try:
        status["state"] = "training_direction"
        status["message"] = "Training Direction Model..."
        print("\n[Train] Starting Direction Model training...", flush=True)

        from models.direction_trainer import train
        result = train(warm_start=False)

        status["direction_f1"] = result["f1_weighted"]
        status["state"] = "idle"
        status["message"] = f"Direction Model trained. F1={result['f1_weighted']:.4f}"
        print(f"[Train] ✅ Direction Model done. F1={result['f1_weighted']:.4f}", flush=True)

    except Exception as e:
        status["state"] = "error"
        status["message"] = f"Direction training failed: {str(e)}"
        status["last_error"] = str(e)
        print(f"[Train] ❌ Direction training failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        operation_lock.release()


def run_train_range():
    """Train the Range Model (HIGH + LOW prediction)."""
    global status

    if not operation_lock.acquire(blocking=False):
        status["message"] = "Another operation is already running."
        return

    try:
        status["state"] = "training_range"
        status["message"] = "Training Range Model (HIGH + LOW)..."
        print("\n[Train] Starting Range Model training...", flush=True)

        from models.range_trainer import train
        result = train(warm_start=False)

        status["range_mae_high"] = result["mae_high"]
        status["range_mae_low"] = result["mae_low"]
        status["state"] = "idle"
        status["message"] = (f"Range Model trained. "
                             f"HIGH MAE=${result['mae_high']:.2f}, "
                             f"LOW MAE=${result['mae_low']:.2f}")
        print(f"[Train] ✅ Range Model done. "
              f"HIGH MAE=${result['mae_high']:.2f}, LOW MAE=${result['mae_low']:.2f}", flush=True)

    except Exception as e:
        status["state"] = "error"
        status["message"] = f"Range training failed: {str(e)}"
        status["last_error"] = str(e)
        print(f"[Train] ❌ Range training failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        operation_lock.release()


def run_start_scheduler():
    """Start the 15-min heartbeat scheduler."""
    global status
    try:
        from scheduler.job_runner import start_scheduler
        start_scheduler()
        status["message"] = "Scheduler running (15-min heartbeat)."
        print("[Scheduler] ✅ Started.", flush=True)
    except Exception as e:
        status["last_error"] = str(e)
        print(f"[Scheduler] ❌ Failed: {e}", flush=True)


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server starts immediately. Self-ping keeps it alive."""
    print("\n[START] Server binding to port...", flush=True)

    # Start self-ping thread to keep Render alive
    ping_thread = threading.Thread(target=self_ping_loop, daemon=True)
    ping_thread.start()
    print("[START] Self-ping thread started (every 5 min).", flush=True)
    print("[START] ✅ Server is LIVE. Visit dashboard to control bot.", flush=True)

    yield

    print("[EXIT] Server shutting down.", flush=True)


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="BTC Trading Bot",
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static files
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health endpoint — keeps Render alive."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/status")
def get_status():
    """Returns current bot status for the dashboard."""
    return JSONResponse(content=status)


@app.post("/api/fetch-data")
def trigger_fetch():
    """Trigger data fetch (runs in background thread)."""
    if status["state"] not in ("idle", "error", "complete"):
        return JSONResponse({"error": "Operation already running"}, status_code=409)
    threading.Thread(target=run_fetch_data, daemon=True).start()
    return {"message": "Data fetch started. Check /api/status for progress."}


@app.post("/api/train-direction")
def trigger_train_direction():
    """Trigger Direction Model training (runs in background thread)."""
    if status["state"] not in ("idle", "error", "complete"):
        return JSONResponse({"error": "Operation already running"}, status_code=409)
    threading.Thread(target=run_train_direction, daemon=True).start()
    return {"message": "Direction training started. Check /api/status for progress."}


@app.post("/api/train-range")
def trigger_train_range():
    """Trigger Range Model training (runs in background thread)."""
    if status["state"] not in ("idle", "error", "complete"):
        return JSONResponse({"error": "Operation already running"}, status_code=409)
    threading.Thread(target=run_train_range, daemon=True).start()
    return {"message": "Range training started. Check /api/status for progress."}


@app.post("/api/start-scheduler")
def trigger_scheduler():
    """Start the 15-min prediction scheduler."""
    threading.Thread(target=run_start_scheduler, daemon=True).start()
    return {"message": "Scheduler starting."}


@app.post("/api/reset-tables")
def trigger_reset():
    """Drop all tables and recreate (DANGEROUS — use only for fresh start)."""
    if status["state"] not in ("idle", "error"):
        return JSONResponse({"error": "Operation running, can't reset"}, status_code=409)

    try:
        from data.database import get_connection, create_tables
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DROP TABLE IF EXISTS candles CASCADE;
                    DROP TABLE IF EXISTS trades CASCADE;
                    DROP TABLE IF EXISTS model_store CASCADE;
                    DROP TABLE IF EXISTS scalers CASCADE;
                    DROP TABLE IF EXISTS app_logs CASCADE;
                    DROP TABLE IF EXISTS portfolio CASCADE;
                    DROP TABLE IF EXISTS predictions_log CASCADE;
                """)
            conn.commit()
        finally:
            conn.close()
        create_tables()
        status["state"] = "idle"
        status["message"] = "Tables reset. Ready for fresh start."
        status["candles_fetched"] = 0
        status["direction_f1"] = None
        status["range_mae_high"] = None
        status["range_mae_low"] = None
        return {"message": "All tables dropped and recreated."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def serve_dashboard():
    """Serve the control dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ── Inline Dashboard HTML ─────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BTC Trading Bot — Control Panel</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 24px; }
    h1 { font-size: 1.5rem; margin-bottom: 8px; color: #818cf8; }
    .subtitle { color: #64748b; margin-bottom: 32px; font-size: 0.9rem; }
    .status-card { background: #1e2030; border: 1px solid #2d3348; border-radius: 12px; padding: 20px; margin-bottom: 24px; }
    .status-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .status-value { font-size: 1.1rem; font-weight: 600; }
    .status-value.idle { color: #94a3b8; }
    .status-value.fetching, .status-value.training_direction, .status-value.training_range { color: #fbbf24; }
    .status-value.complete { color: #22c55e; }
    .status-value.error { color: #ef4444; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }
    .metric { background: #161825; border-radius: 8px; padding: 12px; text-align: center; }
    .metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; }
    .metric-value { font-size: 1.2rem; font-weight: 700; margin-top: 4px; color: #e2e8f0; }
    .buttons { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 24px; }
    button { padding: 14px 20px; border: none; border-radius: 8px; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .btn-fetch { background: #3b82f6; color: white; }
    .btn-fetch:hover { background: #2563eb; }
    .btn-direction { background: #8b5cf6; color: white; }
    .btn-direction:hover { background: #7c3aed; }
    .btn-range { background: #06b6d4; color: white; }
    .btn-range:hover { background: #0891b2; }
    .btn-scheduler { background: #22c55e; color: white; }
    .btn-scheduler:hover { background: #16a34a; }
    .btn-reset { background: #ef4444; color: white; }
    .btn-reset:hover { background: #dc2626; }
    .log { background: #161825; border-radius: 8px; padding: 16px; margin-top: 24px; font-family: monospace; font-size: 0.8rem; color: #94a3b8; max-height: 200px; overflow-y: auto; white-space: pre-wrap; }
    .refresh-note { color: #64748b; font-size: 0.75rem; margin-top: 12px; text-align: center; }
  </style>
</head>
<body>
  <h1>⚡ BTC Trading Bot — Control Panel</h1>
  <p class="subtitle">15-min candles | Dual AI models (Direction + Range) | BTC/USDT spot</p>

  <div class="status-card">
    <div class="status-label">Current Status</div>
    <div class="status-value" id="state">Loading...</div>
    <div style="color:#94a3b8; margin-top:8px; font-size:0.85rem;" id="message"></div>
    <div class="metrics">
      <div class="metric">
        <div class="metric-label">Candles in DB</div>
        <div class="metric-value" id="candles">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">Direction F1</div>
        <div class="metric-value" id="f1">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">Range HIGH MAE</div>
        <div class="metric-value" id="mae-high">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">Range LOW MAE</div>
        <div class="metric-value" id="mae-low">—</div>
      </div>
    </div>
  </div>

  <div class="buttons">
    <button class="btn-fetch" onclick="doAction('/api/fetch-data')">📥 Fetch / Resume Data</button>
    <button class="btn-direction" onclick="doAction('/api/train-direction')">🧠 Train Direction Model</button>
    <button class="btn-range" onclick="doAction('/api/train-range')">📈 Train Range Model</button>
    <button class="btn-scheduler" onclick="doAction('/api/start-scheduler')">⏱️ Start Scheduler</button>
    <button class="btn-reset" onclick="if(confirm('This will DELETE all data. Are you sure?')) doAction('/api/reset-tables')">🗑️ Reset All Tables</button>
  </div>

  <div class="log" id="log">Waiting for status...</div>
  <div class="refresh-note">Status refreshes every 3 seconds</div>

  <script>
    const logEl = document.getElementById('log');
    let logLines = [];

    function addLog(msg) {
      const time = new Date().toLocaleTimeString();
      logLines.push(`[${time}] ${msg}`);
      if (logLines.length > 50) logLines.shift();
      logEl.textContent = logLines.join('\\n');
      logEl.scrollTop = logEl.scrollHeight;
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const stateEl = document.getElementById('state');
        stateEl.textContent = data.state.toUpperCase();
        stateEl.className = 'status-value ' + data.state;
        document.getElementById('message').textContent = data.message || '';
        document.getElementById('candles').textContent = data.candles_fetched ? data.candles_fetched.toLocaleString() : '—';
        document.getElementById('f1').textContent = data.direction_f1 ? data.direction_f1.toFixed(4) : '—';
        document.getElementById('mae-high').textContent = data.range_mae_high ? '$' + data.range_mae_high.toFixed(2) : '—';
        document.getElementById('mae-low').textContent = data.range_mae_low ? '$' + data.range_mae_low.toFixed(2) : '—';
      } catch(e) {
        document.getElementById('state').textContent = 'OFFLINE';
      }
    }

    async function doAction(endpoint) {
      addLog(`Triggering ${endpoint}...`);
      try {
        const res = await fetch(endpoint, { method: 'POST' });
        const data = await res.json();
        addLog(data.message || data.error || 'Done');
      } catch(e) {
        addLog(`Error: ${e.message}`);
      }
    }

    fetchStatus();
    setInterval(fetchStatus, 3000);
  </script>
</body>
</html>"""


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
