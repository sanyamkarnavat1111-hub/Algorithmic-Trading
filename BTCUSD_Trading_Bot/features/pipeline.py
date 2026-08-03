"""
features/pipeline.py — Feature engineering pipeline for both AI models.

Builds features from raw OHLCV data and generates labels for training:
  - Direction labels: BUY (2) / HOLD (1) / SELL (0)
  - Range labels: predicted HIGH and LOW across next 5 candles

Used by both the Direction Model trainer and the Range Model trainer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import psycopg2.extras

from features.indicators import add_indicators
from features.time_features import add_time_features
from data.database import get_connection
from config import LABEL_LOOKAHEAD, DIRECTION_THRESHOLD


# ── Feature columns the models use as input ───────────────────────────────────

FEATURE_COLUMNS = [
    # Momentum
    "rsi_14", "stoch_k", "stoch_d",
    # Trend
    "macd", "macd_signal", "macd_hist",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "adx",
    # Volatility
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_position",
    "atr_14",
    # Volume
    "obv", "vwap",
    # Price action
    "candle_body", "candle_range", "body_to_range_ratio",
    "close_to_ema21", "close_to_ema200",
    # Time (cyclic)
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    # Raw price context
    "volume", "close",
]


# ── Label Generation ──────────────────────────────────────────────────────────

def generate_direction_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate BUY/HOLD/SELL labels based on 5-candle-ahead return.

    Logic:
      - If price goes UP more than 0.3% in next 5 candles → BUY (2)
      - If price goes DOWN more than 0.3% in next 5 candles → SELL (0)
      - Otherwise → HOLD (1)

    Drops the last 5 rows (no future data available for them).
    """
    df = df.copy()
    future_close = df["close"].shift(-LABEL_LOOKAHEAD)
    future_return = (future_close - df["close"]) / df["close"]

    df["direction_label"] = 1  # HOLD by default
    df.loc[future_return > DIRECTION_THRESHOLD, "direction_label"] = 2   # BUY
    df.loc[future_return < -DIRECTION_THRESHOLD, "direction_label"] = 0  # SELL

    # Remove rows where we don't have future data
    df = df.iloc[:-LABEL_LOOKAHEAD].copy()
    return df


def generate_range_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate HIGH and LOW labels for the Range Model.

    For each candle, looks at the next 5 candles and finds:
      - range_high_label: the MAXIMUM high price across those 5 candles
      - range_low_label: the MINIMUM low price across those 5 candles

    Drops the last 5 rows (no future data available for them).
    """
    df = df.copy()

    # Rolling max of HIGH over next 5 candles (shift to look forward)
    # We reverse, do rolling max, then reverse back
    high_reversed = df["high"].iloc[::-1]
    low_reversed = df["low"].iloc[::-1]

    # Rolling window of 5 on reversed series = "next 5" on original
    rolling_high = high_reversed.rolling(window=LABEL_LOOKAHEAD, min_periods=LABEL_LOOKAHEAD).max()
    rolling_low = low_reversed.rolling(window=LABEL_LOOKAHEAD, min_periods=LABEL_LOOKAHEAD).min()

    # Reverse back and shift by 1 (we want NEXT 5 candles, not including current)
    df["range_high_label"] = rolling_high.iloc[::-1].shift(-1).values
    df["range_low_label"] = rolling_low.iloc[::-1].shift(-1).values

    # Remove rows where we don't have future data
    df = df.dropna(subset=["range_high_label", "range_low_label"])
    df = df.iloc[:-LABEL_LOOKAHEAD].copy()
    return df


# ── Scaler Persistence ────────────────────────────────────────────────────────

def _ensure_scalers_table():
    """Create scalers table if it doesn't exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scalers (
                    scaler_id VARCHAR(30) PRIMARY KEY,
                    scaler_blob BYTEA NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_scaler(scaler: StandardScaler, scaler_id: str):
    """Save a fitted scaler to PostgreSQL."""
    _ensure_scalers_table()
    buf = io.BytesIO()
    joblib.dump(scaler, buf)
    buf.seek(0)
    blob = buf.read()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scalers (scaler_id, scaler_blob)
                VALUES (%s, %s)
                ON CONFLICT (scaler_id) DO UPDATE
                  SET scaler_blob = EXCLUDED.scaler_blob,
                      updated_at = NOW()
            """, (scaler_id, psycopg2.Binary(blob)))
        conn.commit()
    finally:
        conn.close()


def load_scaler(scaler_id: str):
    """Load a previously saved scaler from PostgreSQL."""
    _ensure_scalers_table()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT scaler_blob FROM scalers WHERE scaler_id = %s", (scaler_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    buf = io.BytesIO(bytes(row[0]))
    return joblib.load(buf)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, fit_scaler: bool = False,
                   scaler_id: str = "default") -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
      1. Compute technical indicators (RSI, MACD, EMAs, etc.)
      2. Add cyclic time features (hour, day, month sin/cos)
      3. Select only the feature columns we need
      4. Scale features using StandardScaler

    Args:
        df: Raw OHLCV DataFrame with columns [open_time, open, high, low, close, volume]
        fit_scaler: True during training (fits new scaler), False during inference
        scaler_id: Identifier for saving/loading the scaler

    Returns:
        DataFrame with scaled features + metadata columns (open_time, close_raw, high_raw, low_raw)
    """
    if df.empty or len(df) < 250:
        raise ValueError(f"Not enough data: got {len(df)} rows, need 250+")

    # Step 1: Technical indicators
    df = add_indicators(df)

    # Step 2: Cyclic time features
    df = add_time_features(df)

    # Step 3: Select feature columns
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    feature_df = df[available].copy()

    # Step 4: Scale features
    if fit_scaler:
        scaler = StandardScaler()
        feature_df[available] = scaler.fit_transform(feature_df[available])
        save_scaler(scaler, scaler_id)
        print(f"[Pipeline] Fitted and saved scaler '{scaler_id}'.")
    else:
        scaler = load_scaler(scaler_id)
        if scaler is None:
            raise RuntimeError(f"No scaler found for '{scaler_id}'. Run training first.")
        feature_df[available] = scaler.transform(feature_df[available])

    # Add metadata columns (unscaled, for reference)
    feature_df["open_time"] = df["open_time"].values
    feature_df["close_raw"] = df["close"].values
    feature_df["high_raw"] = df["high"].values
    feature_df["low_raw"] = df["low"].values

    return feature_df


def get_feature_columns() -> list:
    """Return the list of feature column names used by models."""
    return FEATURE_COLUMNS
