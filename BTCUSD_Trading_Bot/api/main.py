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
        sync_timeframe("15m", progress_callback=log)

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
    """Manually start the scheduler (fallback if auto-start didn't trigger)."""
    global status
    try:
        from scheduler.job_runner import start_scheduler, heartbeat
        start_scheduler()
        status["scheduler_running"] = True
        log("✅ Scheduler started manually.")
        heartbeat()
        log("First heartbeat done.")
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
    log("Server started.")
    threading.Thread(target=self_ping_loop, daemon=True).start()
    log("Self-ping active (keeps Render alive).")

    # Auto-check if models + data exist and start trading
    threading.Thread(target=_auto_start_trading, daemon=True).start()

    yield
    print("[EXIT] Server stopped.", flush=True)


def _auto_start_trading():
    """Check if data + models are ready. If yes, start scheduler + first trade immediately."""
    import time as _time
    _time.sleep(5)

    try:
        from models.registry import get_model_info
        from data.database import get_connection, create_tables

        # Ensure tables exist and sequences are synced (fixes duplicate key errors)
        create_tables()

        # Check candle data
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM candles WHERE timeframe = '15m'")
                candle_count = cur.fetchone()[0]
        finally:
            conn.close()

        if candle_count < 250:
            log(f"⚠️ Not enough data ({candle_count} candles, need 250+). Use /admin to fetch data first.")
            return

        # Check all 3 models
        direction = get_model_info("15m", "direction")
        range_high = get_model_info("15m", "range_high")
        range_low = get_model_info("15m", "range_low")

        missing = []
        if direction["version"] is None:
            missing.append("Direction")
        if range_high["version"] is None:
            missing.append("Range HIGH")
        if range_low["version"] is None:
            missing.append("Range LOW")

        if missing:
            log(f"⚠️ Models not trained: {', '.join(missing)}. Use /admin to train them.")
            return

        # Everything ready — start trading
        log(f"✅ All systems ready. Data: {candle_count:,} candles. Models: all trained.")

        from scheduler.job_runner import start_scheduler, heartbeat
        start_scheduler()
        status["scheduler_running"] = True
        log("Scheduler started (every 15 min).")

        # First trade immediately
        log("Running first prediction + trade...")
        heartbeat()
        log("First heartbeat done. Bot is LIVE! 🚀")

    except Exception as e:
        log(f"❌ Auto-start error: {e}")
        import traceback
        traceback.print_exc()


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
    """Returns current state + model info. Called by Check Status button."""
    from models.registry import get_model_info

    direction_info = get_model_info("15m", "direction")
    range_high_info = get_model_info("15m", "range_high")
    range_low_info = get_model_info("15m", "range_low")

    ready_for_trading = (
        direction_info["version"] is not None and
        range_high_info["version"] is not None and
        range_low_info["version"] is not None
    )

    return JSONResponse({
        "state": status["state"],
        "scheduler_running": status["scheduler_running"],
        "ready_for_trading": ready_for_trading,
        "direction_model": direction_info,
        "range_high_model": range_high_info,
        "range_low_model": range_low_info,
    })


@app.get("/api/latest-candle")
def get_latest_candle():
    """Returns the most recent candle stored in the database."""
    try:
        from data.database import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT open_time, open, high, low, close, volume
                    FROM candles WHERE timeframe = '15m'
                    ORDER BY open_time DESC LIMIT 1
                """)
                row = cur.fetchone()
                cur.execute("SELECT COUNT(*) FROM candles WHERE timeframe = '15m'")
                count = cur.fetchone()[0]
        finally:
            conn.close()

        if row:
            return {
                "total_candles": count,
                "latest": {
                    "open_time": row[0].isoformat() if row[0] else None,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            }
        return {"total_candles": 0, "latest": None}
    except Exception as e:
        return {"error": str(e)}


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


@app.post("/api/clear-trades")
def trigger_clear_trades():
    """Clear trades, portfolio, and prediction logs only. Keeps candles and models."""
    try:
        from data.database import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trades")
                cur.execute("DELETE FROM portfolio")
                cur.execute("DELETE FROM predictions_log")
                cur.execute("DELETE FROM app_logs")
            conn.commit()
        finally:
            conn.close()
        log("🧹 Cleared trades, portfolio, and logs. Fresh start.")
        return {"ok": True, "msg": "Trades cleared. Portfolio will reinitialize on next heartbeat."}
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)


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
  <title>Bot Admin</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0b0d13; color: #e2e8f0; min-height: 100vh; padding: 24px; max-width: 700px; margin: 0 auto; }
    h1 { font-size: 1.3rem; color: #818cf8; margin-bottom: 24px; }
    .actions { display: flex; flex-direction: column; gap: 10px; margin-bottom: 24px; }
    .btn { padding: 14px 18px; border: none; border-radius: 8px; font-size: 0.9rem; font-weight: 600; cursor: pointer; color: white; text-align: left; }
    .btn-blue { background: #1d4ed8; }
    .btn-purple { background: #6d28d9; }
    .btn-cyan { background: #0e7490; }
    .btn-green { background: #15803d; }
    .btn-red { background: #991b1b; }
    .btn-gray { background: #334155; }
    .result { background: #1a1d2e; border: 1px solid #2d3348; border-radius: 8px; padding: 16px; margin-top: 16px; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; min-height: 60px; }
  </style>
</head>
<body>
  <h1>⚡ BTC Trading Bot — Admin</h1>

  <div class="actions">
    <button class="btn btn-gray" onclick="checkLatest()">🔍 Check Latest Candle in DB</button>
    <button class="btn btn-gray" onclick="checkStatus()">📊 Check Training & Trading Status</button>
    <button class="btn btn-blue" onclick="doAction('fetch-data')">📥 Fetch / Resume Data</button>
    <button class="btn btn-purple" onclick="doAction('train-direction')">🧠 Train Direction Model</button>
    <button class="btn btn-cyan" onclick="doAction('train-range')">📈 Train Range Model</button>
    <button class="btn btn-red" onclick="if(confirm('DELETE all data?')) doAction('reset-tables')">🗑️ Reset Everything</button>
    <button class="btn btn-gray" onclick="doAction('clear-trades')">🧹 Clear Trades Only (keep data + models)</button>
  </div>

  <div class="result" id="result">Click a button above to see results here.</div>

  <script>
    const resultEl = document.getElementById('result');

    async function checkLatest() {
      resultEl.textContent = 'Loading...';
      try {
        const res = await fetch('/api/latest-candle');
        const data = await res.json();
        if (data.latest) {
          resultEl.textContent =
            'Total candles in DB: ' + data.total_candles.toLocaleString() + '\\n\\n' +
            'Latest candle:\\n' +
            '  Time:   ' + data.latest.open_time + '\\n' +
            '  Open:   $' + data.latest.open.toLocaleString() + '\\n' +
            '  High:   $' + data.latest.high.toLocaleString() + '\\n' +
            '  Low:    $' + data.latest.low.toLocaleString() + '\\n' +
            '  Close:  $' + data.latest.close.toLocaleString() + '\\n' +
            '  Volume: ' + data.latest.volume.toLocaleString();
        } else if (data.error) {
          resultEl.textContent = 'Error: ' + data.error;
        } else {
          resultEl.textContent = 'No candles in database yet.';
        }
      } catch(e) {
        resultEl.textContent = 'Failed to connect: ' + e.message;
      }
    }

    async function checkStatus() {
      resultEl.textContent = 'Loading...';
      try {
        const res = await fetch('/api/admin-status');
        const data = await res.json();
        let text = '';
        text += 'Current state: ' + data.state.toUpperCase() + '\\n';
        text += 'Scheduler running: ' + (data.scheduler_running ? 'YES' : 'NO') + '\\n';
        text += '\\n';

        text += '--- Direction Model (BUY/SELL/HOLD) ---\\n';
        if (data.direction_model.version) {
          text += '  Status:    TRAINED ✅\\n';
          text += '  Version:   v' + data.direction_model.version + '\\n';
          text += '  F1 Score:  ' + (data.direction_model.accuracy || '—') + '\\n';
          text += '  Trained:   ' + (data.direction_model.trained_at || '—') + '\\n';
          text += '  Rows used: ' + (data.direction_model.train_rows ? data.direction_model.train_rows.toLocaleString() : '—') + '\\n';
        } else {
          text += '  Status: NOT TRAINED ❌\\n';
        }
        text += '\\n';

        text += '--- Range HIGH Model ---\\n';
        if (data.range_high_model.version) {
          text += '  Status:    TRAINED ✅\\n';
          text += '  Version:   v' + data.range_high_model.version + '\\n';
          text += '  MAE:       $' + (data.range_high_model.accuracy || '—') + '\\n';
          text += '  Trained:   ' + (data.range_high_model.trained_at || '—') + '\\n';
        } else {
          text += '  Status: NOT TRAINED ❌\\n';
        }
        text += '\\n';

        text += '--- Range LOW Model ---\\n';
        if (data.range_low_model.version) {
          text += '  Status:    TRAINED ✅\\n';
          text += '  Version:   v' + data.range_low_model.version + '\\n';
          text += '  MAE:       $' + (data.range_low_model.accuracy || '—') + '\\n';
          text += '  Trained:   ' + (data.range_low_model.trained_at || '—') + '\\n';
        } else {
          text += '  Status: NOT TRAINED ❌\\n';
        }
        text += '\\n';

        if (data.ready_for_trading) {
          text += '🟢 READY FOR LIVE TRADING — all models trained.\\n';
          text += '   Trading starts automatically on server boot.\\n';
          text += '   Scheduler: ' + (data.scheduler_running ? 'RUNNING ✅' : 'NOT RUNNING (restart server)');
        } else {
          text += '🔴 NOT READY — train all models first.\\n';
          text += '   Once all 3 models are trained, trading starts automatically on next deploy.';
        }

        resultEl.textContent = text;
      } catch(e) {
        resultEl.textContent = 'Failed to connect: ' + e.message;
      }
    }

    async function doAction(action) {
      resultEl.textContent = 'Triggered: ' + action + '\\nRunning in background. Click "Check Latest Candle" to monitor progress.';
      try {
        const res = await fetch('/api/' + action, { method: 'POST' });
        const data = await res.json();
        if (data.ok === false) {
          resultEl.textContent = 'Blocked: ' + (data.msg || 'Another operation is running.');
        }
      } catch(e) {
        resultEl.textContent = 'Error: ' + e.message;
      }
    }
  </script>
</body>
</html>"""


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
