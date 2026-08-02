"""
data/database.py — PostgreSQL connection and table creation.
Pure PostgreSQL implementation using psycopg2.
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
    """Create all tables if they don't already exist. Safe to run multiple times."""
    sql = """
    -- Raw OHLCV candle data from Binance
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

    -- Computed features for each candle (stored as JSONB for flexibility)
    CREATE TABLE IF NOT EXISTS features (
        id           SERIAL PRIMARY KEY,
        timeframe    VARCHAR(10) NOT NULL,
        open_time    TIMESTAMP   NOT NULL,
        feature_data JSONB       NOT NULL,
        label        SMALLINT,           -- 0=SELL, 1=HOLD, 2=BUY
        created_at   TIMESTAMP   DEFAULT NOW(),
        UNIQUE (timeframe, open_time)
    );

    -- All paper trades made by any AI model
    CREATE TABLE IF NOT EXISTS trades (
        id            SERIAL PRIMARY KEY,
        model_id      VARCHAR(20)   NOT NULL,
        signal        VARCHAR(10)   NOT NULL,
        confidence    NUMERIC(5,4)  NOT NULL,
        entry_price   NUMERIC(20,8) NOT NULL,
        exit_price    NUMERIC(20,8),
        stop_loss     NUMERIC(20,8) NOT NULL,
        position_size NUMERIC(20,8) NOT NULL,
        pnl           NUMERIC(20,8),
        pnl_pct       NUMERIC(10,6),
        status        VARCHAR(20)   NOT NULL DEFAULT 'OPEN',
        opened_at     TIMESTAMP     DEFAULT NOW(),
        closed_at     TIMESTAMP
    );

    -- Model version history — tracks every retrain
    CREATE TABLE IF NOT EXISTS model_versions (
        id           SERIAL PRIMARY KEY,
        model_id     VARCHAR(20)  NOT NULL,
        version      INTEGER      NOT NULL,
        trained_at   TIMESTAMP    DEFAULT NOW(),
        train_rows   INTEGER,
        accuracy     NUMERIC(6,4),
        is_active    BOOLEAN      DEFAULT FALSE
    );

    -- Application logs (only warnings/errors + key events, cleaned after retrain)
    CREATE TABLE IF NOT EXISTS app_logs (
        id         SERIAL PRIMARY KEY,
        level      VARCHAR(10)  NOT NULL,  -- INFO, WARNING, ERROR
        model_id   VARCHAR(20),
        message    TEXT         NOT NULL,
        created_at TIMESTAMP    DEFAULT NOW()
    );
    """

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("[DB] All PostgreSQL tables created / verified.")
    finally:
        conn.close()


def log_event(level: str, message: str, model_id: str = None):
    """Write a log entry to PostgreSQL."""
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


def clean_old_logs(days_to_keep: int = 7):
    """Delete logs older than `days_to_keep` days."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM app_logs
                WHERE level = 'INFO'
                  AND created_at < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))
            cur.execute("""
                DELETE FROM app_logs
                WHERE level IN ('WARNING', 'ERROR')
                  AND created_at < NOW() - INTERVAL '30 days'
            """)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    create_tables()
