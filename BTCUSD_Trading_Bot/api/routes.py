"""
api/routes.py — Dashboard API endpoints.

Provides data for the frontend:
  - /api/dashboard — predictions, positions, P&L stats
  - /api/candles/{timeframe} — OHLCV data for charting
  - /api/models/status — model health check
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from data.binance_fetcher import load_candles
from models.predictor import predict
from models.registry import get_model_info
from data.database import get_connection
from config import TIMEFRAMES

router = APIRouter()


@router.get("/api/dashboard")
def get_dashboard():
    """Returns all data needed for the UI dashboard."""
    try:
        btc_price = _get_latest_btc_price()

        models_data = []
        for timeframe in TIMEFRAMES.keys():
            model_id = TIMEFRAMES[timeframe]["model_id"]

            prediction = predict(timeframe)
            direction_info = get_model_info(timeframe, "direction")
            range_high_info = get_model_info(timeframe, "range_high")
            range_low_info = get_model_info(timeframe, "range_low")

            open_positions = _get_open_positions(model_id)
            activity_log = _get_activity_log(model_id)
            stats = _get_position_stats(model_id)

            models_data.append({
                "model_id": model_id,
                "timeframe": timeframe,
                "prediction": prediction,
                "direction_model": direction_info,
                "range_high_model": range_high_info,
                "range_low_model": range_low_info,
                "open_positions": open_positions,
                "activity_log": activity_log,
                "stats": stats,
            })

        return JSONResponse({
            "models": models_data,
            "btc_price": btc_price,
        })
    except Exception as e:
        import traceback
        print(f"Dashboard error: {e}\n{traceback.format_exc()}", flush=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/candles/{timeframe}")
def get_candles(timeframe: str, limit: int = 100):
    """Return recent OHLCV candles for charting."""
    if timeframe not in TIMEFRAMES:
        return {"error": f"Unknown timeframe: {timeframe}"}

    df = load_candles(timeframe, limit=limit)
    if df.empty:
        return {"candles": []}

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "t": row["open_time"].isoformat(),
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row["volume"]),
        })
    return {"candles": candles, "timeframe": timeframe}


@router.get("/api/models/status")
def get_models_status():
    """Quick status check for all models."""
    statuses = {}
    for timeframe in TIMEFRAMES.keys():
        model_id = TIMEFRAMES[timeframe]["model_id"]
        statuses[model_id] = {
            "direction": get_model_info(timeframe, "direction"),
            "range_high": get_model_info(timeframe, "range_high"),
            "range_low": get_model_info(timeframe, "range_low"),
        }
    return statuses


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_latest_btc_price() -> float:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT close FROM candles
                WHERE timeframe = '15m'
                ORDER BY open_time DESC LIMIT 1
            """)
            row = cur.fetchone()
    finally:
        conn.close()
    return float(row[0]) if row else 0.0


def _get_open_positions(model_id: str) -> list:
    """Fetch currently open positions for this model."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, entry_price, amount_usdt, btc_quantity, 
                       predicted_high, predicted_low, opened_at
                FROM positions
                WHERE model_id = %s AND status = 'OPEN'
                ORDER BY opened_at ASC
            """, (model_id,))
            
            positions = []
            for row in cur.fetchall():
                positions.append({
                    "id": row[0],
                    "entry_price": float(row[1]),
                    "amount_usdt": float(row[2]),
                    "btc_quantity": float(row[3]),
                    "predicted_high": float(row[4]),
                    "predicted_low": float(row[5]),
                    "opened_at": row[6].isoformat() if row[6] else None
                })
            return positions
    finally:
        conn.close()

def _get_activity_log(model_id: str) -> list:
    """Fetch the latest 50 ACTION logs."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT message, created_at
                FROM app_logs
                WHERE model_id = %s AND level = 'ACTION'
                ORDER BY created_at DESC LIMIT 50
            """, (model_id,))
            
            logs = []
            for row in cur.fetchall():
                logs.append({
                    "message": row[0],
                    "timestamp": row[1].isoformat() if row[1] else None
                })
            return logs
    finally:
        conn.close()


def _get_position_stats(model_id: str) -> dict:
    """Get overall trading stats."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    COALESCE(SUM(pnl), 0) as total_pnl
                FROM positions
                WHERE model_id = %s AND status = 'CLOSED'
            """, (model_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    total = row[0] or 0
    wins = row[1] or 0
    return {
        "total_trades": total,
        "wins": wins,
        "losses": row[2] or 0,
        "win_rate": round(wins / total, 4) if total > 0 else 0,
        "total_pnl": round(float(row[3]), 2),
    }
