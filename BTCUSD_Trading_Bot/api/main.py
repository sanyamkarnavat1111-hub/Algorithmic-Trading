"""
api/main.py — FastAPI application entry point.

Routes:
  /       → Trading dashboard (Phase 2 — shows predictions, portfolio, charts)
  /admin  → Setup & control panel (fetch data, train models, live logs)
  /health → Keep-alive endpoint for Render

Server starts immediately (no blocking). All heavy operations run in background.
Self-ping keeps Render alive.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from collections import deque
import threading
import time
import traceback

import requests as http_requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from config import API_HOST, API_PORT

UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")

# ── Global State & Live Logs ──────────────────────────────────────────────────

# Status tracks what the bot is doing right now
status = {
    "state": "idle",
    "candles_fetched": 0,
    "direction_f1": None,
    "range_mae_high": None,
    "range_mae_low": None,
    "scheduler_running": False,
}

# Live log buffer (last 200 lines — polled by dashboard every 3 seconds)
live_logs = deque(maxlen=200)
operation_lock = threading.Lock()


def log(msg: str):
    """Add a timestamped message to the live log buffer AND print to stdout."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    live_logs.append(line)
    print(line, flush=True)


# ── Self-Ping ─────────────────────────────────────────────────────────────────

def self_ping_loop():
    """Pings /health every 5 min to keep Render awake."""
    time.sleep(30)
    while True:
        try:
            http_requests.get(f"http://localhost:{API_PORT}/health", timeout=5)
        except Exception:
            pass
        time.sleep(300)


# ── Background Operations ─────────────────────────────────────────────────────

def run_fetch_data():
    """Fetch 15-min candles. Resumes from last candle (incremental)."""
    global status
    if not operation_lock.acquire(blocking=False):
        log("⚠️ Another operation is running. Try again later.")
        return
    try:
        status["state"] = "fetching"
        log("━━━ STARTING DATA FETCH ━━━")

        # Ensure tables exist
        from data.database import create_tables
        create_tables()
        log("Tables verified.")

        from data.binance_fetcher import test_binance_connection, sync_timeframe

        if not test_binance_connection():
            log("❌ Cannot reach Binance API.")
            status["state"] = "error"
            return

        log("Connected to Binance. Fetching 15-min candles (resumes from last saved)...")
        sync_timeframe("15m")

        # Count total candles
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
        log(f"✅ Data fetch complete. {count:,} candles in database.")
        log("━━━ FETCH DONE ━━━")

    except Exception as e:
        status["state"] = "error"
        log(f"❌ Fetch failed: {e}")
        traceback.print_exc()
    finally:
        operation_lock.release()


def run_train_direction():
    """Train Direction Model (BUY/SELL/HOLD classification)."""
    global status
    if not operation_lock.acquire(blocking=False):
        log("⚠️ Another operation is running.")
        return
    try:
        status["state"] = "training_direction"
        log("━━━ TRAINING DIRECTION MODEL ━━━")
        log("This may take a few minutes...")

        from models.direction_trainer import train
        result = train(warm_start=False)

        status["direction_f1"] = result["f1_weighted"]
        status["state"] = "idle"
        log(f"✅ Direction Model trained!")
        log(f"   F1 Weighted: {result['f1_weighted']:.4f} (target ≥ 0.47)")
        log(f"   F1 Macro: {result['f1_macro']:.4f}")
        log(f"   Accuracy: {result['accuracy']:.4f}")
        log(f"   CV Mean F1: {result['cv_mean_f1']:.4f}")
        log(f"   Train rows: {result['train_rows']:,}")
        log(f"   Version: v{result['version']}")
        log("━━━ DIRECTION TRAINING DONE ━━━")

    except Exception as e:
        status["state"] = "error"
        log(f"❌ Direction training failed: {e}")
        traceback.print_exc()
    finally:
        operation_lock.release()


def run_train_range():
    """Train Range Model (HIGH + LOW regression)."""
    global status
    if not operation_lock.acquire(blocking=False):
        log("⚠️ Another operation is running.")
        return
    try:
        status["state"] = "training_range"
        log("━━━ TRAINING RANGE MODEL ━━━")
        log("Training HIGH and LOW predictors...")

        from models.range_trainer import train
        result = train(warm_start=False)

        status["range_mae_high"] = result["mae_high"]
        status["range_mae_low"] = result["mae_low"]
        status["state"] = "idle"
        log(f"✅ Range Model trained!")
        log(f"   HIGH — MAE: ${result['mae_high']:,.2f} | RMSE: ${result['rmse_high']:,.2f} | R²: {result['r2_high']:.4f}")
        log(f"   LOW  — MAE: ${result['mae_low']:,.2f} | RMSE: ${result['rmse_low']:,.2f} | R²: {result['r2_low']:.4f}")
        log(f"   HIGH error as % of price: {result['pct_error_high']:.3f}%")
        log(f"   LOW error as % of price: {result['pct_error_low']:.3f}%")
        log("━━━ RANGE TRAINING DONE ━━━")

    except Exception as e:
        status["state"] = "error"
        log(f"❌ Range training failed: {e}")
        traceback.print_exc()
    finally:
        operation_lock.release()


def run_start_scheduler():
    """Start the 15-min trading scheduler."""
    global status
    try:
        from scheduler.job_runner import start_scheduler
        start_scheduler()
        status["scheduler_running"] = True
        log("✅ Scheduler started! Bot will trade every 15 minutes.")
    except Exception as e:
        log(f"❌ Scheduler failed: {e}")


def run_reset_tables():
    """Drop and recreate all tables."""
    global status
    if not operation_lock.acquire(blocking=False):
        log("⚠️ Another operation is running.")
        return
    try:
        from data.database import get_connection, create_tables
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DROP TABLE IF EXISTS candles, trades, model_store, scalers,
                    app_logs, portfolio, predictions_log CASCADE;
                """)
            conn.commit()
        finally:
            conn.close()
        create_tables()
        status["state"] = "idle"
        status["candles_fetched"] = 0
        status["direction_f1"] = None
        status["range_mae_high"] = None
        status["range_mae_low"] = None
        log("🗑️ All tables dropped and recreated. Fresh start.")
    except Exception as e:
        log(f"❌ Reset failed: {e}")
    finally:
        operation_lock.release()


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[START] Server binding to port...", flush=True)
    log("Server started. Visit /admin to control the bot.")
    threading.Thread(target=self_ping_loop, daemon=True).start()
    log("Self-ping active (keeps Render alive).")
    yield
    print("[EXIT] Server stopped.", flush=True)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="BTC Trading Bot", version="2.0.0", lifespan=lifespan)

from api.routes import router
app.include_router(router)

if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/api/admin-status")
def admin_status():
    """Single endpoint the admin dashboard polls for ALL info."""
    return JSONResponse({
        **status,
        "logs": list(live_logs),
    })


@app.post("/api/fetch-data")
def trigger_fetch():
    if status["state"] not in ("idle", "error"):
        return JSONResponse({"ok": False, "msg": "Busy"}, status_code=409)
    threading.Thread(target=run_fetch_data, daemon=True).start()
    return {"ok": True}


@app.post("/api/train-direction")
def trigger_train_direction():
    if status["state"] not in ("idle", "error"):
        return JSONResponse({"ok": False, "msg": "Busy"}, status_code=409)
    threading.Thread(target=run_train_direction, daemon=True).start()
    return {"ok": True}


@app.post("/api/train-range")
def trigger_train_range():
    if status["state"] not in ("idle", "error"):
        return JSONResponse({"ok": False, "msg": "Busy"}, status_code=409)
    threading.Thread(target=run_train_range, daemon=True).start()
    return {"ok": True}


@app.post("/api/start-scheduler")
def trigger_scheduler():
    threading.Thread(target=run_start_scheduler, daemon=True).start()
    return {"ok": True}


@app.post("/api/reset-tables")
def trigger_reset():
    if status["state"] not in ("idle", "error"):
        return JSONResponse({"ok": False, "msg": "Busy"}, status_code=409)
    threading.Thread(target=run_reset_tables, daemon=True).start()
    return {"ok": True}


# ── Trading Dashboard (/) ─────────────────────────────────────────────────────

@app.get("/")
def serve_trading_dashboard():
    """Main trading dashboard (Phase 2)."""
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Trading dashboard coming soon. Visit <a href='/admin'>/admin</a> for setup.</h2>")


# ── Admin Dashboard (/admin) ──────────────────────────────────────────────────

@app.get("/admin")
def serve_admin():
    return HTMLResponse(content=ADMIN_HTML)


ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bot Admin — Setup & Monitor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0b0d13; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; }

    /* Header */
    .header { background: #12141d; border-bottom: 1px solid #1e2235; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
    .header h1 { font-size: 1.1rem; color: #818cf8; }
    .header .badge { font-size: 0.7rem; padding: 4px 10px; border-radius: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .badge-idle { background: #1e293b; color: #64748b; }
    .badge-fetching { background: #422006; color: #fbbf24; }
    .badge-training_direction, .badge-training_range { background: #1e1b4b; color: #a78bfa; }
    .badge-error { background: #450a0a; color: #f87171; }

    /* Main layout */
    .main { display: flex; flex: 1; overflow: hidden; }

    /* Left panel: controls + metrics */
    .panel { width: 320px; background: #12141d; border-right: 1px solid #1e2235; padding: 20px; overflow-y: auto; }
    .section-title { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; margin-top: 20px; }
    .section-title:first-child { margin-top: 0; }

    /* Metrics */
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .metric { background: #1a1d2e; border-radius: 8px; padding: 10px; text-align: center; }
    .metric-label { font-size: 0.6rem; color: #64748b; text-transform: uppercase; }
    .metric-value { font-size: 1rem; font-weight: 700; margin-top: 2px; }

    /* Buttons */
    .actions { display: flex; flex-direction: column; gap: 8px; }
    .btn { padding: 12px 16px; border: none; border-radius: 8px; font-size: 0.85rem; font-weight: 600; cursor: pointer; text-align: left; transition: all 0.15s; display: flex; align-items: center; gap: 8px; }
    .btn:hover { transform: translateX(2px); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .btn-blue { background: #1d4ed8; color: white; }
    .btn-purple { background: #6d28d9; color: white; }
    .btn-cyan { background: #0e7490; color: white; }
    .btn-green { background: #15803d; color: white; }
    .btn-red { background: #991b1b; color: white; }

    /* Right panel: live logs */
    .logs-panel { flex: 1; display: flex; flex-direction: column; padding: 20px; }
    .logs-header { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; display: flex; justify-content: space-between; }
    .logs { flex: 1; background: #0f1119; border: 1px solid #1e2235; border-radius: 8px; padding: 14px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.75rem; line-height: 1.6; overflow-y: auto; color: #94a3b8; white-space: pre-wrap; word-break: break-word; }
    .logs .log-line { margin-bottom: 2px; }
    .logs .log-success { color: #4ade80; }
    .logs .log-error { color: #f87171; }
    .logs .log-warn { color: #fbbf24; }
    .logs .log-header { color: #818cf8; font-weight: 600; }

    @media (max-width: 768px) {
      .main { flex-direction: column; }
      .panel { width: 100%; border-right: none; border-bottom: 1px solid #1e2235; }
      .logs-panel { min-height: 400px; }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>⚡ BTC Trading Bot — Admin</h1>
    <span class="badge badge-idle" id="badge">IDLE</span>
  </div>

  <div class="main">
    <!-- Left: Controls -->
    <div class="panel">
      <div class="section-title">System Metrics</div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Candles</div>
          <div class="metric-value" id="m-candles">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Direction F1</div>
          <div class="metric-value" id="m-f1">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">HIGH MAE</div>
          <div class="metric-value" id="m-high">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">LOW MAE</div>
          <div class="metric-value" id="m-low">—</div>
        </div>
      </div>

      <div class="section-title">Actions</div>
      <div class="actions">
        <button class="btn btn-blue" id="btn-fetch" onclick="doAction('fetch-data')">📥 Fetch / Resume Data</button>
        <button class="btn btn-purple" id="btn-dir" onclick="doAction('train-direction')">🧠 Train Direction Model</button>
        <button class="btn btn-cyan" id="btn-range" onclick="doAction('train-range')">📈 Train Range Model</button>
        <button class="btn btn-green" id="btn-sched" onclick="doAction('start-scheduler')">⏱️ Start Live Trading</button>
        <button class="btn btn-red" id="btn-reset" onclick="if(confirm('DELETE all data?')) doAction('reset-tables')">🗑️ Reset Everything</button>
      </div>

      <div class="section-title">How to Use</div>
      <div style="font-size:0.75rem; color:#64748b; line-height:1.5;">
        1. Click <b>Fetch Data</b> — downloads BTC candles<br>
        2. Click <b>Train Direction</b> — trains BUY/SELL/HOLD model<br>
        3. Click <b>Train Range</b> — trains HIGH/LOW predictor<br>
        4. Click <b>Start Live Trading</b> — bot trades every 15 min<br><br>
        If server restarts, just click Fetch again (it resumes).
      </div>
    </div>

    <!-- Right: Live Logs -->
    <div class="logs-panel">
      <div class="logs-header">
        <span>Live Logs (auto-refresh 3s)</span>
        <span id="log-count">0 lines</span>
      </div>
      <div class="logs" id="logs"></div>
    </div>
  </div>

  <script>
    const logsEl = document.getElementById('logs');
    let prevLogCount = 0;

    async function poll() {
      try {
        const res = await fetch('/api/admin-status');
        const data = await res.json();

        // Badge
        const badge = document.getElementById('badge');
        badge.textContent = data.state.toUpperCase().replace('_', ' ');
        badge.className = 'badge badge-' + data.state;

        // Metrics
        document.getElementById('m-candles').textContent = data.candles_fetched ? data.candles_fetched.toLocaleString() : '—';
        document.getElementById('m-f1').textContent = data.direction_f1 ? data.direction_f1.toFixed(4) : '—';
        document.getElementById('m-high').textContent = data.range_mae_high ? '$' + data.range_mae_high.toFixed(0) : '—';
        document.getElementById('m-low').textContent = data.range_mae_low ? '$' + data.range_mae_low.toFixed(0) : '—';

        // Disable buttons when busy
        const busy = !['idle', 'error'].includes(data.state);
        document.getElementById('btn-fetch').disabled = busy;
        document.getElementById('btn-dir').disabled = busy;
        document.getElementById('btn-range').disabled = busy;
        document.getElementById('btn-reset').disabled = busy;

        // Logs
        if (data.logs && data.logs.length !== prevLogCount) {
          prevLogCount = data.logs.length;
          logsEl.innerHTML = data.logs.map(line => {
            let cls = '';
            if (line.includes('✅')) cls = 'log-success';
            else if (line.includes('❌')) cls = 'log-error';
            else if (line.includes('⚠️')) cls = 'log-warn';
            else if (line.includes('━━━')) cls = 'log-header';
            return `<div class="log-line ${cls}">${escHtml(line)}</div>`;
          }).join('');
          logsEl.scrollTop = logsEl.scrollHeight;
          document.getElementById('log-count').textContent = data.logs.length + ' lines';
        }
      } catch(e) {
        document.getElementById('badge').textContent = 'OFFLINE';
        document.getElementById('badge').className = 'badge badge-error';
      }
    }

    function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

    async function doAction(action) {
      try { await fetch('/api/' + action, { method: 'POST' }); } catch(e) {}
    }

    poll();
    setInterval(poll, 3000);
  </script>
</body>
</html>"""


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
