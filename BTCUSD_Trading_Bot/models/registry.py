"""
models/registry.py — Save, load, and version models in PostgreSQL.

Supports multiple model types per timeframe:
  - "direction" — classification model (BUY/SELL/HOLD)
  - "range_high" — regression model (predicted HIGH)
  - "range_low" — regression model (predicted LOW)

Each model type is versioned independently.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from data.database import get_connection


def _ensure_models_table():
    """Create the model storage table if it doesn't exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_store (
                    id          SERIAL PRIMARY KEY,
                    timeframe   VARCHAR(10)  NOT NULL,
                    model_type  VARCHAR(20)  NOT NULL,
                    version     INTEGER      NOT NULL,
                    model_blob  BYTEA        NOT NULL,
                    accuracy    NUMERIC(10,4),
                    train_rows  INTEGER,
                    trained_at  TIMESTAMP    DEFAULT NOW(),
                    is_active   BOOLEAN      DEFAULT FALSE,
                    UNIQUE(timeframe, model_type, version)
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_model(model, timeframe: str, model_type: str,
               accuracy: float, train_rows: int) -> int:
    """
    Save a trained LightGBM model to PostgreSQL.
    Marks it as active, deactivating the previous version.

    Args:
        model: trained LightGBM Booster
        timeframe: "15m"
        model_type: "direction", "range_high", or "range_low"
        accuracy: primary metric (F1 for direction, MAE for range)
        train_rows: number of training samples used

    Returns: version number
    """
    _ensure_models_table()

    model_str = model.model_to_string()
    model_bytes = model_str.encode("utf-8")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Get next version number for this timeframe + model_type
            cur.execute("""
                SELECT COALESCE(MAX(version), 0) + 1
                FROM model_store
                WHERE timeframe = %s AND model_type = %s
            """, (timeframe, model_type))
            version = cur.fetchone()[0]

            # Deactivate previous active model
            cur.execute("""
                UPDATE model_store SET is_active = FALSE
                WHERE timeframe = %s AND model_type = %s
            """, (timeframe, model_type))

            # Insert new model as active
            cur.execute("""
                INSERT INTO model_store
                  (timeframe, model_type, version, model_blob, accuracy, train_rows, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            """, (timeframe, model_type, version,
                  psycopg2.Binary(model_bytes), accuracy, train_rows))

        conn.commit()
        print(f"[Registry] Saved {model_type} model v{version} for {timeframe}")
        return version
    finally:
        conn.close()


def load_active_model(timeframe: str, model_type: str):
    """
    Load the currently active model for a timeframe and type.

    Returns LightGBM Booster or None if no model exists.
    """
    _ensure_models_table()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_blob FROM model_store
                WHERE timeframe = %s AND model_type = %s AND is_active = TRUE
                ORDER BY trained_at DESC LIMIT 1
            """, (timeframe, model_type))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    model_bytes = bytes(row[0])
    model_str = model_bytes.decode("utf-8")

    import lightgbm as lgb
    model = lgb.Booster(model_str=model_str)
    return model


def get_model_info(timeframe: str, model_type: str) -> dict:
    """
    Return metadata about the active model.

    Returns dict with: version, accuracy, train_rows, trained_at
    """
    _ensure_models_table()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT version, accuracy, train_rows, trained_at
                FROM model_store
                WHERE timeframe = %s AND model_type = %s AND is_active = TRUE
                ORDER BY trained_at DESC LIMIT 1
            """, (timeframe, model_type))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"version": None, "accuracy": None, "train_rows": None, "trained_at": None}

    return {
        "version": row[0],
        "accuracy": float(row[1]) if row[1] else None,
        "train_rows": row[2],
        "trained_at": row[3].isoformat() if row[3] else None,
    }
