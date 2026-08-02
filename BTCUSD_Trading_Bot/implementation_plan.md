# Crypto Trading Bot — Updated MVP Implementation Plan

> **Folder**: `Crypto-Trading-Bot/`  
> **Market**: BTC/USDT on Binance API (free, no ccxt)  
> **Style**: Swing Trading (1–5 day holds)  
> **Capital**: Unlimited paper trading (so AI never "dies" while learning)

---

## What We Are Building

Three parallel AI models, each focused on a different timeframe, all paper-trading BTC/USDT simultaneously. A clean web dashboard shows what all three models are doing in real time. The system self-improves by retraining every 50–100 trades using the outcomes it generated.

```
Binance API (Historical + Live OHLCV)
         │
         ▼
PostgreSQL (Render) ──→ Feature Pipeline (auto normalize/scale)
         │
         ├──→ AI Model #1 (1-Hour Candles)  → Paper Trades → Retraining Loop
         ├──→ AI Model #2 (8-Hour Candles)  → Paper Trades → Retraining Loop
         └──→ AI Model #3 (1-Day Candles)   → Paper Trades → Retraining Loop
                                                    │
                                                    ▼
                                         Clean Web Dashboard (UI)
                                         + /health endpoint (cronjob)
```

---

## Confirmed Decisions

| Decision | Choice | Reason |
|---|---|---|
| ML Model | **LightGBM** | Low RAM, fast, warm-start retraining, ideal for Render free tier |
| Market | **BTC/USDT** | Free Binance API, 24/7 data, most liquid crypto |
| Data Source | **Binance API** (direct, no ccxt) | Free, simple, reliable |
| Database | **PostgreSQL on Render** | 1 GB enough for months of testing |
| Trading Style | **Swing (1–5 days)** | Clean signals, manageable for ML |
| Capital | **Unlimited (paper trading)** | AI trains without risk of crashing |
| Position Size | **2% per trade (fixed)** | Simple, MVP-safe |
| Loss Function | **Standard Cross-Entropy** | Simple, works well for classification |
| Timeframes | **1h, 8h, 1d (parallel)** | Three AI models, each with own rhythm |
| Deployment | **Render (free tier)** | Backend + UI hosted together |
| Live Trading | **Skipped for now** | Paper trading focus for MVP |
| Prediction Horizon | **Next 5 candles** | Broader than 1-candle, not too noisy |
| Trade Check Frequency | **Every 15 minutes** | Check all models, enter/exit if signal fires |

> [!NOTE]
> **On prediction horizon**: You suggested 20-30 candles. For 1h candles that is 20-30 hours ahead — good for swing. For 8h that is 7-10 days and for 1d that is 20-30 days, which becomes too speculative. I recommend **next 5 candles** as the label horizon for all three models. This gives the 1h model a 5-hour view, 8h model a 40-hour view, and 1d model a 5-day view — perfectly aligned with swing trading. We can tune this later.

---

## The Three AI Models

| Model | Candle Size | Predicts | Checks For Trade |
|---|---|---|---|
| **AI-1** | 1-Hour | BUY/SELL/HOLD 5 candles ahead (5h) | Every 15 minutes |
| **AI-2** | 8-Hour | BUY/SELL/HOLD 5 candles ahead (40h) | Every 15 minutes |
| **AI-3** | 1-Day | BUY/SELL/HOLD 5 candles ahead (5 days) | Every 15 minutes |

All three run **independently** and log their own trades. The dashboard shows them side by side.

> [!TIP]
> Even though we check every 15 minutes, the AI only acts when a new candle has closed on its timeframe. So 1h AI acts hourly, 8h AI acts every 8 hours, 1d AI acts daily. The 15-min heartbeat just ensures no signal is missed.

---

## Phase Roadmap

| Phase | What We Build | Timeline |
|---|---|---|
| **Phase 1 (MVP — This Plan)** | Data pipeline + 3 LightGBM models + paper trading + dashboard | 8 Weeks |
| **Phase 2** | Add River for true online/streaming ML (per-candle updates) | Post-MVP |
| **Phase 3** | Add LSTM / Transformer layers for sequence modeling | Post-MVP |
| **Phase 4** | Add RL (PPO/SAC) for dynamic position sizing | Post-MVP |
| **Phase 5** | Add News Sentiment / macro signals (Fed rate months, etc.) | Post-MVP |

---

## Feature Engineering Pipeline

This pipeline runs **automatically** every time new data arrives. The same pipeline is used for training AND inference, so there is no mismatch.

### Step 1 — Raw OHLCV Features
```
open, high, low, close, volume
candle_body = close - open
candle_range = high - low
body_to_range_ratio = candle_body / candle_range
```

### Step 2 — Technical Indicators (via pandas-ta)
| Indicator | Parameters | Purpose |
|---|---|---|
| RSI | 14 | Overbought / Oversold |
| MACD | 12, 26, 9 | Trend Momentum |
| Bollinger Bands | 20, 2 | Volatility breakouts |
| EMA | 9, 21, 50, 200 | Trend direction (multiple horizons) |
| ATR | 14 | Volatility — also used for stop-loss sizing |
| OBV | — | Volume confirms price move |
| Stochastic | 14, 3 | Momentum |
| VWAP | — | Fair value anchor |
| ADX | 14 | Trend strength (filters choppy markets) |

### Step 3 — Cyclic Time Features
> You specifically asked for this — so the model can learn patterns like "Fed rate decision months" or "Monday gap-fills".

```python
# Encode cyclical features as sin/cos so AI understands circular nature
hour_sin   = sin(2π × hour / 24)
hour_cos   = cos(2π × hour / 24)
day_sin    = sin(2π × day_of_week / 7)
day_cos    = cos(2π × day_of_week / 7)
month_sin  = sin(2π × month / 12)
month_cos  = cos(2π × month / 12)
```

This lets the model learn: "August historically behaves differently" or "Friday closes often reverse on Monday."

### Step 4 — Normalization & Scaling
- **Price + Volume**: `StandardScaler` (zero mean, unit variance)
- **Oscillators (RSI, Stochastic)**: Already 0–100, divide by 100
- **MACD**: `MinMaxScaler`
- **Scaler is serialized** (saved to disk) so inference uses exact same scaling as training

### Step 5 — Label Generation
```python
future_return = (close[t + 5] - close[t]) / close[t]

if future_return > +0.5%:  label = BUY   (1)
if future_return < -0.5%:  label = SELL  (-1)
else:                       label = HOLD  (0)
```

> The 0.5% threshold filters out noise. We tune this per timeframe later.

---

## Database Schema (PostgreSQL on Render)

```sql
-- Raw market data per timeframe
CREATE TABLE candles (
    id          SERIAL PRIMARY KEY,
    timeframe   VARCHAR(10),      -- '1h', '8h', '1d'
    open_time   TIMESTAMP,
    open        DECIMAL,
    high        DECIMAL,
    low         DECIMAL,
    close       DECIMAL,
    volume      DECIMAL,
    fetched_at  TIMESTAMP DEFAULT NOW()
);

-- All features computed by pipeline
CREATE TABLE features (
    id          SERIAL PRIMARY KEY,
    candle_id   INTEGER REFERENCES candles(id),
    timeframe   VARCHAR(10),
    feature_data JSONB,           -- all indicators stored as JSON
    created_at  TIMESTAMP DEFAULT NOW()
);

-- All paper trades made by any AI model
CREATE TABLE trades (
    id            SERIAL PRIMARY KEY,
    model_id      VARCHAR(20),    -- 'ai_1h', 'ai_8h', 'ai_1d'
    signal        VARCHAR(10),    -- 'BUY', 'SELL', 'HOLD'
    confidence    DECIMAL,        -- model's prediction probability
    entry_price   DECIMAL,
    exit_price    DECIMAL,
    stop_loss     DECIMAL,
    pnl           DECIMAL,        -- filled after trade closes
    status        VARCHAR(20),    -- 'OPEN', 'CLOSED', 'STOPPED_OUT'
    created_at    TIMESTAMP DEFAULT NOW(),
    closed_at     TIMESTAMP
);

-- Model version history for the self-improving loop
CREATE TABLE model_versions (
    id            SERIAL PRIMARY KEY,
    model_id      VARCHAR(20),
    version       INTEGER,
    trained_at    TIMESTAMP,
    accuracy      DECIMAL,
    sharpe_ratio  DECIMAL,
    is_active     BOOLEAN DEFAULT FALSE
);
```

---

## Self-Improving Online Learning Loop

```
Every 50-100 trades (per model):
  ┌─────────────────────────────────────────┐
  │ 1. Pull all closed trades from DB       │
  │ 2. Attach actual P&L as feedback label  │
  │ 3. Retrain LightGBM (warm-start)        │
  │    lgbm.train(init_model=current_model) │
  │ 4. Backtest new model on last 60 days   │
  │ 5. Compare accuracy + Sharpe ratio      │
  │ 6. If better → deploy new model version │
  │    If worse  → keep current model       │
  │ 7. Log to model_versions table          │
  └─────────────────────────────────────────┘
```

Each of the 3 AIs has its own independent retraining loop. They do not share weights.

---

## Risk Management (Non-Negotiable)

1. **Position size**: Fixed 2% of portfolio per trade
2. **Stop-loss**: `entry_price - (1.5 × ATR)` — auto-computed, always set
3. **Max open trades per model**: 1 at a time (simple for MVP)
4. **Confidence filter**: Only trade if prediction probability > **60%**
5. **Daily loss circuit breaker**: If a model loses >5% in 24h → pauses for 24h

---

## Web Dashboard (Clean UI)

The UI shows all three models running in real-time. Looking at it, you should immediately understand what each AI is doing.

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 Crypto Trading Bot — Live Dashboard         [●] LIVE   │
├──────────────┬──────────────┬──────────────────────────────┤
│  AI-1 (1H)   │  AI-2 (8H)   │  AI-3 (1D)                  │
│  Signal: BUY │  Signal: HOLD│  Signal: SELL                │
│  Conf: 73%   │  Conf: 52%   │  Conf: 81%                  │
│  P&L: +2.3%  │  P&L: -0.5% │  P&L: +5.1%                 │
├──────────────┴──────────────┴──────────────────────────────┤
│  BTC/USDT Price Chart (candlestick) with trade markers      │
│  [Entry ▲]  [Exit ▼]  [Stop-Loss ✕]                        │
├─────────────────────────────────────────────────────────────┤
│  Recent Trades Table (all 3 models, color-coded)            │
│  Model | Signal | Entry | Exit | P&L | Status              │
├─────────────────────────────────────────────────────────────┤
│  Model Health: AI-1 v4 ✅  AI-2 v2 ✅  AI-3 v1 ⚠️          │
│  Last Retrain: 2h ago / 6h ago / 22h ago                   │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack for UI
| Component | Technology |
|---|---|
| Backend (API) | **FastAPI** (Python) — clean, fast, easy to read |
| Frontend | **Plain HTML + Chart.js + vanilla JS** — no framework complexity |
| Charts | **Chart.js** (candlestick + trade markers) |
| Real-time updates | **Polling every 30s** (simple, no WebSocket complexity for MVP) |
| Health Endpoint | `GET /health` → returns `{"status": "ok", "timestamp": "..."}` |

> [!IMPORTANT]
> The `/health` endpoint is included. You should set up a **UptimeRobot** or Render cron job to ping it every 14 minutes to keep the free tier server alive.

---

## Codebase Structure

> Every file has one clear job. No over-engineering.

```
Crypto-Trading-Bot/
│
├── data/
│   ├── binance_fetcher.py     # Pulls OHLCV from Binance API (historical + live)
│   └── database.py            # PostgreSQL connection + helper queries
│
├── features/
│   ├── indicators.py          # Computes all technical indicators (pandas-ta)
│   ├── time_features.py       # Cyclic time features (month, day, hour sin/cos)
│   └── pipeline.py            # Normalize + scale + assemble full feature set
│
├── models/
│   ├── trainer.py             # Train LightGBM (initial + warm-start retrain)
│   ├── predictor.py           # Load model → predict BUY/SELL/HOLD + confidence
│   └── registry.py            # Save/load/compare model versions in DB
│
├── trading/
│   ├── paper_trader.py        # Simulates trades, tracks P&L, manages open positions
│   └── risk_manager.py        # 2% sizing, ATR stop-loss, circuit breaker logic
│
├── learning/
│   └── retrain_loop.py        # Triggers retraining every 50-100 trades per model
│
├── scheduler/
│   └── job_runner.py          # APScheduler: runs data fetch + prediction every 15 min
│
├── api/
│   ├── main.py                # FastAPI app — all endpoints including /health
│   └── routes.py              # Dashboard data endpoints
│
├── ui/
│   ├── index.html             # Single-page dashboard
│   ├── charts.js              # Chart.js candlestick + trade markers
│   └── dashboard.js           # Fetches API, updates UI every 30s
│
├── config.py                  # ALL config in one place (API keys, thresholds, DB URL)
├── requirements.txt           # All dependencies
└── README.md                  # How to run the project
```

**Rule**: Each file is under 200 lines. If it grows beyond that, split it.

---

## Deployment on Render (Free Tier)

| Service | What Runs There |
|---|---|
| **Web Service** | FastAPI backend + serves UI + `/health` endpoint |
| **PostgreSQL** | 1 GB database (months of testing data) |
| **Cron Job** | Calls `/health` every 14 min to keep server alive |

> [!WARNING]
> Render free tier sleeps after 15 minutes of inactivity. The `/health` cronjob ping prevents this. Set it up on **UptimeRobot** (free) pointing to your `https://your-app.onrender.com/health`.

### Environment Variables (config.py reads these)
```
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
DATABASE_URL=postgresql://...   ← Render provides this automatically
ENVIRONMENT=production
```

---

## Build Timeline (8 Weeks)

| Week | What We Build | Files Created |
|---|---|---|
| **Week 1** | Binance API data fetcher + PostgreSQL schema | `binance_fetcher.py`, `database.py` |
| **Week 2** | Feature pipeline (indicators + cyclic + scaling) | `indicators.py`, `time_features.py`, `pipeline.py` |
| **Week 3** | LightGBM training + labels + model registry | `trainer.py`, `predictor.py`, `registry.py` |
| **Week 4** | Paper trading + risk manager | `paper_trader.py`, `risk_manager.py` |
| **Week 5** | Scheduler (15-min heartbeat for all 3 AIs) | `job_runner.py` |
| **Week 6** | Self-improving retraining loop | `retrain_loop.py` |
| **Week 7** | FastAPI backend + `/health` + dashboard UI | `main.py`, `routes.py`, `index.html`, `dashboard.js` |
| **Week 8** | Deploy to Render + end-to-end testing | `requirements.txt`, `README.md` |

---

## Open Questions (Resolved)

| Question | Decision |
|---|---|
| Which market? | BTC/USDT on Binance |
| SQLite or PostgreSQL? | PostgreSQL on Render (you provide credentials) |
| Initial capital? | Unlimited paper trading |
| Position size? | 2% per trade (fixed) |
| Trading style? | Swing (1–5 days) |
| How many AIs? | 3 (1h, 8h, 1d timeframes) |
| Prediction horizon? | 5 candles ahead on each timeframe |
| Trade check frequency? | Every 15 minutes |
| UI? | Clean single-page dashboard with Chart.js |
| Health endpoint? | `GET /health` — ping via cronjob |
| Code style? | Clean, readable, max 200 lines per file, clear naming |

---

## What We Are NOT Building (MVP Scope Boundary)

| Feature | Deferred To |
|---|---|
| Live trading with real money | Post-MVP (Phase 2+) |
| News / sentiment analysis | Phase 5 |
| Limited AI budget / 10% monthly return goal | Post-MVP (after paper trading proves the model) |
| RL (PPO/SAC) for dynamic sizing | Phase 4 |
| LSTM / Transformer | Phase 3 |
| Voting ensemble | Phase 3 |
| River (streaming ML) | Phase 2 |
| Order book / Level 2 data | Phase 4+ |

---

## Summary

**Three LightGBM models → 1h / 8h / 1d → BTC/USDT → Paper trade → Learn from outcomes → Retrain → Dashboard shows everything → Deployed on Render.**

That is the complete MVP. Simple, readable, self-improving, deployed.
