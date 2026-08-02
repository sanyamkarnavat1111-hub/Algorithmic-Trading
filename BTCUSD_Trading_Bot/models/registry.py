"""
models/registry.py — Save, load, and compare model versions.
Pure PostgreSQL implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2.extras
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
                    version     INTEGER      NOT NULL,
                    model_blob  BYTEA        NOT NULL,
                    accuracy    NUMERIC(6,4),
                    train_rows  INTEGER,
                    trained_at  TIMESTAMP    DEFAULT NOW(),
                    is_active   BOOLEAN      DEFAULT FALSE,
                    UNIQUE(timeframe, version)
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_model(model, timeframe: str, accuracy: float, train_rows: int) -> int:
    """
    Serialize and save a trained LightGBM model to PostgreSQL.
    Marks it as active (replacing the previous active model).
    """
    _ensure_models_table()

    # Get model string representation
    model_str = model.model_to_string()
    model_bytes = model_str.encode("utf-8")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Get next version number
            cur.execute("""
                SELECT COALESCE(MAX(version), 0) + 1
                FROM model_store
                WHERE timeframe = %s
            """, (timeframe,))
            version = cur.fetchone()[0]

            # Deactivate previous active model
            cur.execute("""
                UPDATE model_store SET is_active = FALSE WHERE timeframe = %s
            """, (timeframe,))

            # Insert new model as active
            cur.execute("""
                INSERT INTO model_store (timeframe, version, model_blob, accuracy, train_rows, is_active)
                VALUES (%s, %s, %s, %s, %s, TRUE)
            """, (timeframe, version, psycopg2.Binary(model_bytes), accuracy, train_rows))

            # Also update model_versions table (used by dashboard)
            cur.execute("""
                INSERT INTO model_versions (model_id, version, accuracy, train_rows, is_active)
                VALUES (%s, %s, %s, %s, TRUE)
                ON CONFLICT DO NOTHING
            """, (f"ai_{timeframe}", version, accuracy, train_rows))

        conn.commit()
        print(f"[Registry] Saved model ai_{timeframe} v{version} (accuracy={accuracy:.4f})")
        return version
    finally:
        conn.close()


def load_active_model(timeframe: str):
    """
    Load the currently active model for a timeframe.
    """
    _ensure_models_table()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_blob FROM model_store
                WHERE timeframe = %s AND is_active = TRUE
                ORDER BY trained_at DESC
                LIMIT 1
            """, (timeframe,))
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


def get_model_info(timeframe: str) -> dict:
    """
    Return metadata about the active model for a timeframe.
    """
    _ensure_models_table()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT version, accuracy, train_rows, trained_at
                FROM model_store
                WHERE timeframe = %s AND is_active = TRUE
                ORDER BY trained_at DESC
                LIMIT 1
            """, (timeframe,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"version": None, "accuracy": None, "train_rows": None, "trained_at": None}

    return {
        "version":    row[0],
        "accuracy":   float(row[1]) if row[1] else None,
        "train_rows": row[2],
        "trained_at": row[3].isoformat() if row[3] else None,
    }
