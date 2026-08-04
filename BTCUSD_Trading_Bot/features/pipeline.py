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

    For each candle at index i, looks at candles i+1 through i+5 and finds:
      - range_high_label: the MAXIMUM high price across those 5 candles
      - range_low_label: the MINIMUM low price across those 5 candles

    Drops the last 5 rows (no future data available for them).
    """
    df = df.copy()

    n = len(df)
    high_labels = np.full(n, np.nan)
    low_labels = np.full(n, np.nan)
    close_prices = df["close"].values

    # For each row, find max high and min low in the NEXT 5 candles as percentage return
    for i in range(n - LABEL_LOOKAHEAD):
        future_slice = df.iloc[i + 1: i + 1 + LABEL_LOOKAHEAD]
        curr_close = close_prices[i]
        high_labels[i] = (future_slice["high"].max() - curr_close) / curr_close
        low_labels[i] = (future_slice["low"].min() - curr_close) / curr_close

    df["range_high_label"] = high_labels
    df["range_low_label"] = low_labels

    # Remove rows where we don't have future data
    df = df.dropna(subset=["range_high_label", "range_low_label"]).reset_index(drop=True)
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
