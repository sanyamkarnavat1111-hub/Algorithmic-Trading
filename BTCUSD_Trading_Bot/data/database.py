"""
data/database.py — PostgreSQL connection and table creation.

Tables:
  - candles: raw 15-min OHLCV data from Binance
  - positions: open and closed trading positions (the core of trading logic)
  - predictions_log: what the models predicted (for accuracy tracking)
  - app_logs: application-level logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import psycopg2.extras
from config import DATABASE_URL


def get_connection():
    """Return a new PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def create_tables():
    """Create all tables if they don't exist. Safe to run multiple times."""
    sql = """
    -- Raw OHLCV candle data from Binance (15-min candles)
    CREATE TABLE IF NOT EXISTS candles (
        id          SERIAL PRIMARY KEY,
        timeframe   VARCHAR(10)   NOT NULL,
        open_time   TIMESTAMP     NOT NULL,
        open        NUMERIC(20,8) NOT NULL,
        high        NUMERIC(20,8) NOT NULL,
        low         NUMERIC(20,8) NOT NULL,
        close       NUMERIC(20,8) NOT NULL,
        volume      NUMERIC(30,8) NOT NULL,
        fetched_at  TIMESTAMP     DEFAULT NOW(),
        UNIQUE (timeframe, open_time)
    );

    -- Trading positions (open and closed)
    -- A position has a clear lifecycle: OPEN → CLOSED
    CREATE TABLE IF NOT EXISTS positions (
        id              SERIAL PRIMARY KEY,
        model_id        VARCHAR(20)   NOT NULL,
        direction       VARCHAR(10)   NOT NULL,
        entry_price     NUMERIC(20,8) NOT NULL,
        exit_price      NUMERIC(20,8),
        amount_usdt     NUMERIC(20,8) NOT NULL,
        btc_quantity    NUMERIC(20,8) NOT NULL,
        predicted_high  NUMERIC(20,8) NOT NULL,
        predicted_low   NUMERIC(20,8) NOT NULL,
        confidence      NUMERIC(5,4)  NOT NULL,
        pnl             NUMERIC(20,8),
        status          VARCHAR(10)   NOT NULL DEFAULT 'OPEN',
        close_reason    VARCHAR(20),
        opened_at       TIMESTAMP     DEFAULT NOW(),
        closed_at       TIMESTAMP
    );

    -- Predictions log (for tracking accuracy over time)
    CREATE TABLE IF NOT EXISTS predictions_log (
        id              SERIAL PRIMARY KEY,
        model_id        VARCHAR(20)   NOT NULL,
        predicted_at    TIMESTAMP     NOT NULL DEFAULT NOW(),
        current_price   NUMERIC(20,8) NOT NULL,
        direction       VARCHAR(10)   NOT NULL,
        confidence      NUMERIC(5,4)  NOT NULL,
        predicted_high  NUMERIC(20,8) NOT NULL,
        predicted_low   NUMERIC(20,8) NOT NULL,
        actual_high     NUMERIC(20,8),
        actual_low      NUMERIC(20,8),
        was_correct     BOOLEAN
    );

    -- Application logs
    CREATE TABLE IF NOT EXISTS app_logs (
        id         SERIAL PRIMARY KEY,
        level      VARCHAR(10)  NOT NULL,
        model_id   VARCHAR(20),
        message    TEXT         NOT NULL,
        created_at TIMESTAMP    DEFAULT NOW()
    );
    """

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)

            # Fix sequences after data migration
            tables_with_serial = ['candles', 'positions', 'predictions_log', 'app_logs']
            for table in tables_with_serial:
                try:
                    cur.execute(f"""
                        SELECT setval(pg_get_serial_sequence('{table}', 'id'),
                                      COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)
                    """)
                except Exception:
                    pass  # Table might not exist yet on first run

        conn.commit()
        print("[DB] All tables created / verified. Sequences synced.")
    finally:
        conn.close()


def log_event(level: str, message: str, model_id: str = None):
    """Write a log entry to PostgreSQL. Silently handles errors."""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO app_logs (level, model_id, message) VALUES (%s, %s, %s)",
                    (level, model_id, message)
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[DB] log_event failed (non-fatal): {e}", flush=True)


def clean_old_logs(days_to_keep: int = 7):
    """Delete logs older than N days."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM app_logs
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    create_tables()
