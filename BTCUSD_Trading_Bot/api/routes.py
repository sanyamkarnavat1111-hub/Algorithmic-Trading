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

        # Get portfolio state
        try:
            from trading.portfolio_manager import get_portfolio
            portfolio = get_portfolio()
        except Exception:
            portfolio = {"usdt_balance": 0, "btc_quantity": 0, "btc_avg_price": 0}

        models_data = []
        for timeframe in TIMEFRAMES.keys():
            model_id = TIMEFRAMES[timeframe]["model_id"]

            prediction = predict(timeframe)
            direction_info = get_model_info(timeframe, "direction")
            range_high_info = get_model_info(timeframe, "range_high")
            range_low_info = get_model_info(timeframe, "range_low")

            # Fetch trades and stats for the UI
            recent_trades, open_trades, stats = _get_trades_and_stats(model_id)

            models_data.append({
                "model_id": model_id,
                "timeframe": timeframe,
                "prediction": prediction,
                "direction_model": direction_info,
                "range_high_model": range_high_info,
                "range_low_model": range_low_info,
                "recent_trades": recent_trades,
                "open_trades": open_trades,
                "stats": stats,
            })

        return JSONResponse({
            "models": models_data,
            "btc_price": btc_price,
            "portfolio": portfolio,
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

def _get_trades_and_stats(model_id: str):
    """Fetch recent trades and basic PnL stats from the trades table."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Get latest 20 trades (excluding HOLD for cleaner view)
            cur.execute("""
                SELECT action, amount_usdt, btc_quantity, price, pnl,
                       confidence, predicted_high, predicted_low, created_at
                FROM trades
                WHERE model_id = %s
                ORDER BY created_at DESC LIMIT 20
            """, (model_id,))

            recent = []
            for row in cur.fetchall():
                recent.append({
                    "signal": row[0],
                    "amount": float(row[1]),
                    "btc_quantity": float(row[2]),
                    "entry_price": float(row[3]),
                    "pnl": float(row[4]) if row[4] else 0.0,
                    "confidence": float(row[5]) if row[5] else None,
                    "predicted_high": float(row[6]) if row[6] else None,
                    "predicted_low": float(row[7]) if row[7] else None,
                    "opened_at": row[8].isoformat() if row[8] else None,
                })

            # Get total P&L (only from BUY/SELL, not HOLD)
            cur.execute("""
                SELECT COALESCE(SUM(pnl), 0) FROM trades
                WHERE model_id = %s AND action IN ('SELL', 'EXIT_TARGET', 'EXIT_RANGE')
            """, (model_id,))
            pnl_row = cur.fetchone()
            total_pnl = float(pnl_row[0]) if pnl_row[0] else 0.0

    finally:
        conn.close()

    stats = {"total_pnl": total_pnl}
    open_trades = []

    return recent, open_trades, stats
