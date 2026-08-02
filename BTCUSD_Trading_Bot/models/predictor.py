"""
models/predictor.py — Load active model and predict BUY/SELL/HOLD.

Takes the latest candle data, runs it through the feature pipeline,
and returns a prediction with confidence score.

Used by the scheduler every 15 minutes for each AI model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data.binance_fetcher import load_candles
from features.pipeline import build_features, get_feature_columns
from models.registry import load_active_model
from config import TIMEFRAMES, MIN_CONFIDENCE

# Map class index back to human-readable signal
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def predict(timeframe: str) -> dict:
    """
    Run a prediction for the given timeframe using the active model.

    Returns a dict:
      {
        "signal":     "BUY" | "SELL" | "HOLD",
        "confidence": 0.0–1.0,
        "should_trade": True/False,   # False if confidence < MIN_CONFIDENCE
        "current_price": float,
        "atr": float,                 # used by risk manager for stop-loss
        "probabilities": {"BUY": x, "SELL": y, "HOLD": z}
      }

    Returns None if model is not trained yet or insufficient data.
    """
    model_id = TIMEFRAMES[timeframe]["model_id"]
    model = load_active_model(timeframe)

    if model is None:
        return None  # Model not trained yet

    # Load enough candles to compute all indicators (need 200+ for EMA-200)
    raw_df = load_candles(timeframe, limit=500)
    if raw_df.empty or len(raw_df) < 250:
        return None

    # Build features (inference mode: fit_scaler=False — use existing scaler)
    feature_df = build_features(raw_df, timeframe, fit_scaler=False)

    if feature_df.empty:
        return None

    # Use only the last row (most recent candle = current market state)
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in feature_df.columns]
    last_row = feature_df[available].iloc[-1:].values

    # Predict: returns shape (1, 3) — probabilities for [SELL, HOLD, BUY]
    proba = model.predict(last_row)[0]
    predicted_class = int(np.argmax(proba))
    confidence = float(proba[predicted_class])
    signal = LABEL_MAP[predicted_class]

    # Current price and ATR from the last raw candle (unscaled)
    current_price = float(raw_df["close"].iloc[-1])
    atr_col = feature_df.get("atr_14") if isinstance(feature_df, pd.DataFrame) else None

    # Get ATR from feature_df (it was scaled, so we compute raw ATR from raw_df)
    # We use the unscaled ATR for stop-loss calculation
    from features.indicators import add_indicators
    raw_with_indicators = add_indicators(raw_df)  # Pass full df (needs 200+ rows for EMA warmup)
    
    if not raw_with_indicators.empty and "atr_14" in raw_with_indicators.columns:
        atr = float(raw_with_indicators["atr_14"].iloc[-1])
    else:
        atr = 0.0

    return {
        "model_id":     model_id,
        "timeframe":    timeframe,
        "signal":       signal,
        "confidence":   round(confidence, 4),
        "should_trade": signal != "HOLD",
        "current_price": current_price,
        "atr":          round(atr, 2),
        "probabilities": {
            "SELL": round(float(proba[0]), 4),
            "HOLD": round(float(proba[1]), 4),
            "BUY":  round(float(proba[2]), 4),
        }
    }
