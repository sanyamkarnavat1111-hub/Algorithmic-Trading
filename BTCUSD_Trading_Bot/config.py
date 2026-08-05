"""
config.py — All configuration in one place.

Reads from environment variables (set in .env for local, Render env vars for production).
Supports local PostgreSQL and Render PostgreSQL.

This bot uses 15-minute candles and two AI models:
  - Direction Model: predicts BUY/SELL/HOLD
  - Range Model: predicts HIGH and LOW for next 5 candles (75 minutes)
"""

import os
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────

parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
local_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))

if os.path.exists(local_env):
    load_dotenv(dotenv_path=local_env, override=True)
    print(f"[Config] Loaded .env from {local_env}")
elif os.path.exists(parent_env):
    load_dotenv(dotenv_path=parent_env, override=True)
    print(f"[Config] Loaded .env from {parent_env}")
else:
    print("[Config] WARNING: No .env file found in local or parent directories.")

# ── Environment & Database ────────────────────────────────────────────────────

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    DATABASE_URL = os.getenv("RENDER_DB_INTERNAL_URL")
else:
    LOCAL_DB_URL = os.getenv("LOCAL_DB_URL")
    DATABASE_URL = LOCAL_DB_URL if LOCAL_DB_URL else os.getenv("RENDER_DB_EXTERNAL_URL")

# Print masked DB URL for verification
if DATABASE_URL:
    try:
        parts = DATABASE_URL.split("@")
        if len(parts) > 1:
            prefix = parts[0].split(":")
            masked = f"{prefix[0]}:{prefix[1]}:*****" if len(prefix) > 2 else f"{prefix[0]}:*****"
            print(f"[Config] Database: {masked}@{parts[1]}")
    except Exception:
        print("[Config] Database URL loaded (masking failed)")
else:
    print("[Config] WARNING: DATABASE_URL is not set!")

# ── Binance API ───────────────────────────────────────────────────────────────

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_BASE_URL = "https://api.binance.com"

# ── Trading Pair ──────────────────────────────────────────────────────────────

SYMBOL = "BTCUSDT"

# ── Timeframe: 15-minute candles ──────────────────────────────────────────────

TIMEFRAMES = {
    "15m": {"interval": "15m", "model_id": "ai_15m"},
}

# ── Prediction Horizon ────────────────────────────────────────────────────────

# Predict 5 candles ahead = 75 minutes into the future
LABEL_LOOKAHEAD = 5

# ── Direction Model (Classification) ─────────────────────────────────────────

# Threshold for labeling: if price moves more than ±0.3% in 75 min → BUY/SELL
# Moves smaller than this are labeled HOLD (noise)
DIRECTION_THRESHOLD = 0.003  # 0.3%

# LightGBM parameters for classification
DIRECTION_LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,              # 0=SELL, 1=HOLD, 2=BUY
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 7,
    "verbose": -1,
}

# No minimum confidence threshold — AI trades on majority vote

# ── Range Model (Regression) ─────────────────────────────────────────────────

# LightGBM parameters for regression (predicting HIGH and LOW)
RANGE_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 7,
    "verbose": -1,
}

# ── Trading Rules ─────────────────────────────────────────────────────────────

# Amount in USDT to spend per trade (bot picks within this range)
TRADE_AMOUNT_MIN = 100   # minimum $100 per trade
TRADE_AMOUNT_MAX = 200   # maximum $200 per trade

# Starting balance (effectively unlimited for paper trading)
STARTING_USDT_BALANCE = 10_000_000

# Multiple positions allowed — no max limit
MAX_OPEN_POSITIONS = 999

# ── Retraining ────────────────────────────────────────────────────────────────

# Retrain both models after this many closed trades
RETRAIN_EVERY_N_TRADES = 100

# ── Scheduler ─────────────────────────────────────────────────────────────────

# Heartbeat interval matches candle timeframe
HEARTBEAT_MINUTES = 15

# ── API Server ────────────────────────────────────────────────────────────────

API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("PORT", 8000))
