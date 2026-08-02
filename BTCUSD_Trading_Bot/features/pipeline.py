"""
features/pipeline.py — Full feature engineering pipeline.
Pure PostgreSQL implementation.
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
from config import LABEL_LOOKAHEAD, BUY_THRESHOLD, SELL_THRESHOLD, LABEL_THRESHOLDS


# ── Columns the model will actually use as features ──────────────────────────
FEATURE_COLUMNS = [
    "rsi_14", "stoch_k", "stoch_d",
    "macd", "macd_signal", "macd_hist",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "adx",
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_position",
    "atr_14",
    "obv", "vwap",
    "candle_body", "candle_range", "body_to_range_ratio",
    "close_to_ema21", "close_to_ema200",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    "volume", "close",
]


# ── Label encoding: 0=SELL, 1=HOLD, 2=BUY ───────────────────────────────────

def _generate_labels(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Generate BUY/HOLD/SELL labels using timeframe-appropriate thresholds."""
    thresholds = LABEL_THRESHOLDS.get(timeframe, {"buy": BUY_THRESHOLD, "sell": SELL_THRESHOLD})
    buy_th  = thresholds["buy"]
    sell_th = thresholds["sell"]

    df = df.copy()
    future_close = df["close"].shift(-LABEL_LOOKAHEAD)
    future_return = (future_close - df["close"]) / df["close"]

    df["label"] = 1  # HOLD by default
    df.loc[future_return > buy_th,  "label"] = 2  # BUY
    df.loc[future_return < sell_th, "label"] = 0  # SELL

    df = df.iloc[:-LABEL_LOOKAHEAD].copy()
    return df


# ── Scaler persistence ────────────────────────────────────────────────────────

def _ensure_scalers_table():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scalers (
                    timeframe VARCHAR(10) PRIMARY KEY,
                    scaler_blob BYTEA NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        conn.close()


def _save_scaler(scaler: StandardScaler, timeframe: str):
    _ensure_scalers_table()
    buf = io.BytesIO()
    joblib.dump(scaler, buf)
    buf.seek(0)
    blob = buf.read()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scalers (timeframe, scaler_blob)
                VALUES (%s, %s)
                ON CONFLICT (timeframe) DO UPDATE
                  SET scaler_blob = EXCLUDED.scaler_blob,
                      updated_at  = NOW()
            """, (timeframe, psycopg2.Binary(blob)))
        conn.commit()
    finally:
        conn.close()


def _load_scaler(timeframe: str):
    _ensure_scalers_table()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT scaler_blob FROM scalers WHERE timeframe = %s
            """, (timeframe,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    buf = io.BytesIO(bytes(row[0]))
    return joblib.load(buf)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, timeframe: str, fit_scaler: bool = False) -> pd.DataFrame:
    if df.empty or len(df) < 250:
        raise ValueError(f"Not enough data for {timeframe}: got {len(df)} rows, need 250+")

    # Step 1 — Technical indicators
    df = add_indicators(df)

    # Step 2 — Cyclic time features
    df = add_time_features(df)

    # Step 3 — Labels (only for training)
    if fit_scaler:
        df = _generate_labels(df, timeframe)

    # Step 4 — Select only the feature columns we want
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    feature_df = df[available].copy()

    # Step 5 — Scale features
    if fit_scaler:
        scaler = StandardScaler()
        feature_df[available] = scaler.fit_transform(feature_df[available])
        _save_scaler(scaler, timeframe)
        print(f"[Pipeline] Fitted and saved scaler for {timeframe}.")
    else:
        scaler = _load_scaler(timeframe)
        if scaler is None:
            raise RuntimeError(f"No scaler found for {timeframe}. Run training first.")
        feature_df[available] = scaler.transform(feature_df[available])

    # Add metadata columns back for reference
    feature_df["open_time"] = df["open_time"].values
    feature_df["close_raw"] = df["close"].values
    if "label" in df.columns:
        feature_df["label"] = df["label"].values

    return feature_df


def get_feature_columns() -> list:
    return FEATURE_COLUMNS
