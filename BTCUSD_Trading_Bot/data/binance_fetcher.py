"""
data/binance_fetcher.py — Fetch OHLCV candle data from Binance REST API.

Supports:
  - Incremental sync: resumes from the last candle in DB (no gaps, no duplicates)
  - Fresh sync: when DB is empty, fetches ALL history from Binance inception (Aug 17 2017)
  - Live update: called by APScheduler every 15 min to stay current

Currently configured for 1H timeframe only.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import pandas as pd
from datetime import datetime, timezone
from data.database import get_connection, log_event
from config import BINANCE_BASE_URL, SYMBOL, TIMEFRAMES


# ── Binance inception date (BTC/USDT launched Aug 17 2017 00:00 UTC) ─────────
BINANCE_INCEPTION_MS = 1502928000000


# ── Low-level API call ────────────────────────────────────────────────────────

def _fetch_klines(interval: str, limit: int = 1000,
                  start_ms: int = None, end_ms: int = None) -> list:
    """
    Call Binance /api/v3/klines. Returns raw list of kline arrays.
    Retries up to 5 times with exponential backoff on network errors.
    """
    url    = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    backoff = 1.0
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == 4:
                print(f"[Binance] Max retries reached. Error: {e}")
                raise
            print(f"[Binance] Network error ({e}). Retry in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2


def _parse_klines(raw_klines: list, timeframe: str) -> pd.DataFrame:
    """Convert raw Binance kline list to a clean DataFrame."""
    rows = []
    for k in raw_klines:
        rows.append({
            "timeframe": timeframe,
            "open_time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open":      float(k[1]),
            "high":      float(k[2]),
            "low":       float(k[3]),
            "close":     float(k[4]),
            "volume":    float(k[5]),
        })
    return pd.DataFrame(rows)


def _save_candles(df: pd.DataFrame, timeframe: str):
    """Insert candles into PostgreSQL. Silently skips duplicates."""
    if df.empty:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO candles (timeframe, open_time, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timeframe, open_time) DO NOTHING
                """, (timeframe, row["open_time"], row["open"],
                      row["high"], row["low"], row["close"], row["volume"]))
        conn.commit()
    finally:
        conn.close()


# ── Public functions ──────────────────────────────────────────────────────────

def sync_timeframe(timeframe: str):
    """
    Incremental synchronization for a given timeframe.

    - If DB has data: resumes from the last stored candle's timestamp + 1ms.
    - If DB is empty: starts from Binance inception date (Aug 17 2017).
    - Paginates forward in batches of 1000 until current time.
    - Safe to call repeatedly — ON CONFLICT skips duplicates.
    """
    interval = TIMEFRAMES[timeframe]["interval"]

    # Find latest candle in DB
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(open_time) FROM candles WHERE timeframe = %s", (timeframe,))
            last_time = cur.fetchone()[0]
    finally:
        conn.close()

    if last_time:
        start_ms = int(last_time.replace(tzinfo=timezone.utc).timestamp() * 1000) + 1
        print(f"[Binance] Resuming {timeframe} from: {last_time} (UTC)")
    else:
        start_ms = BINANCE_INCEPTION_MS
        print(f"[Binance] No data for {timeframe}. Fetching full history from Aug 17 2017...")

    fetched_total = 0
    now_ms = int(time.time() * 1000)

    while start_ms < now_ms:
        raw = _fetch_klines(interval, limit=1000, start_ms=start_ms)

        if not raw:
            break

        df = _parse_klines(raw, timeframe)
        _save_candles(df, timeframe)
        fetched_total += len(raw)

        last_open_ms = raw[-1][0]
        print(f"[Binance] {timeframe}: +{len(raw)} candles (run total: {fetched_total})"
              f" | up to {df['open_time'].iloc[-1]}")

        start_ms = last_open_ms + 1
        if start_ms >= now_ms:
            break

        time.sleep(0.3)

    log_event("INFO", f"Synced {fetched_total} candles", model_id=f"ai_{timeframe}")
    print(f"[Binance] Done. {fetched_total} new candles saved for {timeframe}.")


def fetch_latest(timeframe: str):
    """Called by APScheduler every 15 minutes to keep data current."""
    sync_timeframe(timeframe)


def clear_database_tables():
    """Truncate all data tables. Used with --clean flag."""
    print("[DB] Clearing candles, features, trades, and logs...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE candles, features, trades, app_logs CASCADE;")
        conn.commit()
        print("[DB] All tables cleared.")
    except Exception as e:
        print(f"[DB] Error clearing tables: {e}")
        conn.rollback()
    finally:
        conn.close()


def load_candles(timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """
    Load the most recent `limit` candles from DB, sorted oldest -> newest.
    Used by the feature pipeline and model trainer.
    Pass limit=None to load all candles.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if limit is not None:
                cur.execute("""
                    SELECT open_time, open, high, low, close, volume
                    FROM candles WHERE timeframe = %s
                    ORDER BY open_time DESC LIMIT %s
                """, (timeframe, limit))
            else:
                cur.execute("""
                    SELECT open_time, open, high, low, close, volume
                    FROM candles WHERE timeframe = %s
                    ORDER BY open_time DESC
                """, (timeframe,))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("open_time").reset_index(drop=True)
    df[["open", "high", "low", "close", "volume"]] = \
        df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync BTC/USDT 1H candles from Binance. Safe to re-run — resumes from last candle.")
    parser.add_argument("--clean", action="store_true",
                        help="Truncate all data tables before syncing (fresh start).")
    args = parser.parse_args()

    if args.clean:
        clear_database_tables()

    print("\n--- Syncing 1H Timeframe ---")
    sync_timeframe("1h")
