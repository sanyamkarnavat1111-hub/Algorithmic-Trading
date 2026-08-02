"""
config.py — All configuration in one place.
Reads from environment variables (set in .env for local, Render env vars for production).
Supports local PostgreSQL and Render PostgreSQL.
"""

import os
from dotenv import load_dotenv

# Search paths for .env
parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
local_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))

# Load environment files (override=True ensures local changes take precedence)
if os.path.exists(local_env):
    load_dotenv(dotenv_path=local_env, override=True)
    print(f"[Config] Loaded .env from {local_env}")
elif os.path.exists(parent_env):
    load_dotenv(dotenv_path=parent_env, override=True)
    print(f"[Config] Loaded .env from {parent_env}")
else:
    print("[Config] WARNING: No .env file found in local or parent directories.")

# ── Environment & PostgreSQL Database Routing ───────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    # On Render (Production): Use the internal database URL
    DATABASE_URL = os.getenv("RENDER_DB_INTERNAL_URL")
else:
    # Local (Development): Try LOCAL_DB_URL first, fallback to RENDER_DB_EXTERNAL_URL
    LOCAL_DB_URL = os.getenv("LOCAL_DB_URL")
    if LOCAL_DB_URL:
        DATABASE_URL = LOCAL_DB_URL
    else:
        DATABASE_URL = os.getenv("RENDER_DB_EXTERNAL_URL")

# Mask password for printing
if DATABASE_URL:
    try:
        # Example: postgresql://username:password@host:port/db
        parts = DATABASE_URL.split("@")
        if len(parts) > 1:
            prefix = parts[0].split(":")
            # Mask the password
            masked_prefix = f"{prefix[0]}:{prefix[1]}:*****" if len(prefix) > 2 else f"{prefix[0]}:*****"
            print(f"[Config] Resolved Database URL: {masked_prefix}@{parts[1]}")
        else:
            print(f"[Config] Resolved Database URL: {DATABASE_URL}")
    except Exception:
        print("[Config] Resolved Database URL: [Invalid Format or Masking Error]")
else:
    print("[Config] WARNING: DATABASE_URL is not set!")

# ── Binance API ───────────────────────────────────────────────────────────────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_BASE_URL   = "https://api.binance.com"

# ── Trading Pair ──────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"

# ── Timeframe — 1H only (8h and 1d will be added later) ─────────────────────
TIMEFRAMES = {
    "1h": {"interval": "1h", "model_id": "ai_1h"},
}

# ── Feature Engineering ───────────────────────────────────────────────────────
LABEL_LOOKAHEAD    = 5        # predict 5 candles ahead

# Thresholds per timeframe — wider at longer horizons because moves are larger
# 1h: ±0.5% over next 5 hours is meaningful swing signal
# 8h: ±1.5% over next 40 hours (1.5 days) captures swing moves
# 1d: ±3.0% over next 5 days captures medium-term trend moves
LABEL_THRESHOLDS = {
    "1h": {"buy": 0.005,  "sell": -0.005},
    "8h": {"buy": 0.015,  "sell": -0.015},
    "1d": {"buy": 0.030,  "sell": -0.030},
}

# Keep legacy single values for any code still using them
BUY_THRESHOLD      = 0.005
SELL_THRESHOLD     = -0.005

# ── Model ─────────────────────────────────────────────────────────────────────
MIN_CONFIDENCE     = 0.60    # only trade if model confidence > 60%
RETRAIN_EVERY_N_TRADES = 50  # trigger retraining after N closed trades

LGBM_PARAMS = {
    "objective":     "multiclass",
    "num_class":     3,           # 0=SELL, 1=HOLD, 2=BUY
    "learning_rate": 0.02,        # smaller LR → more trees needed but better generalization
    "num_leaves":    63,          # 63 leaves → good complexity for financial time series
    "max_depth":     7,           # limit depth to prevent overfitting
    "verbose":       -1,
}

# ── Risk Management ───────────────────────────────────────────────────────────
POSITION_SIZE_PCT  = 0.02    # 2% of portfolio per trade
STOP_LOSS_ATR_MULT = 1.5     # stop-loss = entry - (1.5 × ATR)
MAX_OPEN_TRADES    = 1       # max 1 open trade per model at a time
DAILY_LOSS_LIMIT   = 0.05    # pause model if it loses 5% in 24h

# ── Paper Trading ─────────────────────────────────────────────────────────────
PAPER_STARTING_CAPITAL = 1_000_000  # 1 million USDT

# ── Scheduler ─────────────────────────────────────────────────────────────────
HEARTBEAT_MINUTES = 60      # check all models every 15 minutes

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("PORT", 8000))
