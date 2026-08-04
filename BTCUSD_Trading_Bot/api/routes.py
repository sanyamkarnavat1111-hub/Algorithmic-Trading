"""
api/routes.py — Dashboard API endpoints.

Provides data for the frontend:
  - /api/dashboard — all model info, predictions, portfolio state
  - /api/candles/{timeframe} — OHLCV data for charting
  - /api/models/status — quick health check for all models
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

            models_data.append({
                "model_id": model_id,
                "timeframe": timeframe,
                "prediction": prediction,
                "direction_model": direction_info,
                "range_high_model": range_high_info,
                "range_low_model": range_low_info,
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
        return {"error": f"Unknown timeframe: {timeframe}. Use: {list(TIMEFRAMES.keys())}"}

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


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_latest_btc_price() -> float:
    """Get most recent BTC close price from candles table."""
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
