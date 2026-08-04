"""
models/predictor.py — Run both models and return a combined prediction.

Loads the active Direction Model and Range Model, runs features through both,
and returns a unified prediction dict.

Used by the scheduler every 15 minutes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.binance_fetcher import load_candles
from features.pipeline import build_features, get_feature_columns
from models.registry import load_active_model
from config import TIMEFRAMES

# Direction class mapping
DIRECTION_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def predict(timeframe: str) -> dict:
    """
    Run both models and return a combined prediction.

    Returns:
      {
        "direction": "BUY" | "SELL" | "HOLD",
        "confidence": 0.0-1.0,
        "predicted_high": float (dollar price),
        "predicted_low": float (dollar price),
        "current_price": float,
        "should_trade": bool,
        "probabilities": {"BUY": x, "SELL": y, "HOLD": z},
      }

    Returns None if models aren't trained or insufficient data.
    """
    model_id = TIMEFRAMES[timeframe]["model_id"]

    # Load all three models
    direction_model = load_active_model(timeframe, "direction")
    high_model = load_active_model(timeframe, "range_high")
    low_model = load_active_model(timeframe, "range_low")

    if direction_model is None or high_model is None or low_model is None:
        return None  # Models not trained yet

    # Load candles (need 250+ for indicator warmup)
    raw_df = load_candles(timeframe, limit=500)
    if raw_df.empty or len(raw_df) < 250:
        return None

    # Build features using direction scaler (both models use same features)
    feature_df = build_features(raw_df, fit_scaler=False, scaler_id="direction_15m")
    if feature_df.empty:
        return None

    # Get the last row (current market state)
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in feature_df.columns]
    last_row = feature_df[available].iloc[-1:].values

    # Direction prediction
    proba = direction_model.predict(last_row)[0]
    predicted_class = int(np.argmax(proba))
    confidence = float(proba[predicted_class])
    direction = DIRECTION_MAP[predicted_class]

    # Current price
    current_price = float(raw_df["close"].iloc[-1])

    # Range prediction (these predict percentage returns)
    high_pct = float(high_model.predict(last_row)[0])
    low_pct = float(low_model.predict(last_row)[0])

    # Convert back to absolute dollar price for decision engine
    predicted_high = current_price * (1 + high_pct)
    predicted_low = current_price * (1 + low_pct)

    # Should we trade? Just verify direction != HOLD
    should_trade = (direction != "HOLD")

    return {
        "model_id": model_id,
        "timeframe": timeframe,
        "direction": direction,
        "confidence": round(confidence, 4),
        "predicted_high": round(predicted_high, 2),
        "predicted_low": round(predicted_low, 2),
        "current_price": current_price,
        "should_trade": should_trade,
        "probabilities": {
            "SELL": round(float(proba[0]), 4),
            "HOLD": round(float(proba[1]), 4),
            "BUY": round(float(proba[2]), 4),
        },
    }
