"""
api/routes.py — Dashboard data endpoints.

All data the frontend needs to display the 3-model dashboard.
Keeps main.py clean by separating routes here.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter
from data.binance_fetcher import load_candles
from models.predictor import predict
from models.registry import get_model_info
from trading.paper_trader import get_open_trades, get_recent_trades, get_total_pnl
from data.database import get_connection
from config import TIMEFRAMES

router = APIRouter()


@router.get("/api/dashboard")
def get_dashboard():
    """
    Single endpoint that returns everything the dashboard needs.
    Called by the frontend every 30 seconds.

    Returns:
      {
        "models": [ {...model data...}, ... ],
        "btc_price": float
      }
    """
    # Get latest BTC price from DB (fastest source)
    btc_price = _get_latest_btc_price()

    models_data = []
    for timeframe in TIMEFRAMES.keys():
        model_id = TIMEFRAMES[timeframe]["model_id"]

        # Get latest prediction (if model is trained)
        prediction = predict(timeframe)

        # Get model metadata
        info = get_model_info(timeframe)

        # Get trading stats
        stats    = get_total_pnl(model_id)
        portfolio = 1000.0 + stats.get("total_pnl", 0.0) # Base portfolio $1000 + PnL
        open_trades   = get_open_trades(model_id)
        recent_trades = get_recent_trades(model_id, limit=10)

        models_data.append({
            "model_id":    model_id,
            "timeframe":   timeframe,
            "version":     info["version"],
            "accuracy":    info["accuracy"],
            "trained_at":  info["trained_at"],
            "prediction":  prediction,
            "portfolio":   round(portfolio, 2),
            "stats":       stats,
            "open_trades": open_trades,
            "recent_trades": recent_trades,
        })

    return {
        "models":    models_data,
        "btc_price": btc_price,
    }


@router.get("/api/candles/{timeframe}")
def get_candles(timeframe: str, limit: int = 100):
    """
    Return recent OHLCV candles for charting.
    Used by Chart.js candlestick component.

    Args:
        timeframe: '1h', '8h', or '1d'
        limit:     number of candles to return (default 100)
    """
    if timeframe not in TIMEFRAMES:
        return {"error": f"Unknown timeframe: {timeframe}. Use 1h, 8h, or 1d."}

    df = load_candles(timeframe, limit=limit)
    if df.empty:
        return {"candles": []}

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "t":  row["open_time"].isoformat(),
            "o":  float(row["open"]),
            "h":  float(row["high"]),
            "l":  float(row["low"]),
            "c":  float(row["close"]),
            "v":  float(row["volume"]),
        })
    return {"candles": candles, "timeframe": timeframe}


@router.get("/api/trades/{model_id}")
def get_trades(model_id: str, limit: int = 50):
    """Return trade history for a specific model."""
    recent = get_recent_trades(model_id, limit=limit)
    open_t = get_open_trades(model_id)
    stats  = get_total_pnl(model_id)
    return {
        "model_id":     model_id,
        "open_trades":  open_t,
        "closed_trades": recent,
        "stats":        stats,
    }


@router.get("/api/models/status")
def get_models_status():
    """Quick status check for all three models. Used by the health widget."""
    statuses = {}
    for timeframe in TIMEFRAMES.keys():
        model_id = TIMEFRAMES[timeframe]["model_id"]
        info = get_model_info(timeframe)
        statuses[model_id] = {
            "trained": info["version"] is not None,
            "version": info["version"],
            "accuracy": info["accuracy"],
            "trained_at": info["trained_at"],
        }
    return statuses


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_latest_btc_price() -> float:
    """Get most recent BTC close price from candles table (1h is most frequent)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT close FROM candles
                WHERE timeframe = '1h'
                ORDER BY open_time DESC
                LIMIT 1
            """)
            row = cur.fetchone()
    finally:
        conn.close()
    return float(row[0]) if row else 0.0
