# Crypto Trading Bot

Self-learning algorithmic trading bot with 3 parallel AI models on BTC/USDT.

## Architecture

```
Binance API → PostgreSQL → Feature Pipeline → LightGBM Models → Paper Trades → Retrain Loop
                                                                        ↑
                                                               FastAPI Dashboard (UI)
```

## Project Structure

```
Crypto-Trading-Bot/
├── data/
│   ├── binance_fetcher.py   # Fetch OHLCV from Binance API
│   └── database.py          # PostgreSQL tables + logging
├── features/
│   ├── indicators.py        # Technical indicators (pandas-ta)
│   ├── time_features.py     # Cyclic sin/cos time encoding
│   └── pipeline.py          # Full feature pipeline + scaler
├── models/
│   ├── trainer.py           # LightGBM train + warm-start retrain
│   ├── predictor.py         # BUY/SELL/HOLD prediction
│   └── registry.py          # Model versioning in PostgreSQL
├── trading/
│   ├── paper_trader.py      # Paper trading simulator
│   └── risk_manager.py      # 2% sizing, ATR stop-loss, circuit breaker
├── learning/
│   └── retrain_loop.py      # Online learning (triggers after 50 trades)
├── scheduler/
│   └── job_runner.py        # 15-min APScheduler heartbeat
├── api/
│   ├── main.py              # FastAPI app + /health endpoint
│   └── routes.py            # Dashboard API routes
├── ui/
│   ├── index.html           # Dashboard HTML
│   ├── style.css            # Dark mode CSS
│   └── dashboard.js         # Frontend JS (no framework)
├── config.py                # All settings in one place
└── requirements.txt
```

## Setup (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
The `.env` file in the parent directory is loaded automatically.
Required vars:
```
RENDER_DB_EXTERNAL_URL=postgresql://...
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
```

### 3. Fetch historical data (run once)
```bash
python data/binance_fetcher.py
```
This pulls ~1000 candles for 1h, and ~500 for 8h and 1d from Binance.

### 4. Train initial models (run once after data fetch)
```bash
python models/trainer.py
```
Trains LightGBM for all 3 timeframes and saves to PostgreSQL.

### 5. Start the bot
```bash
python api/main.py
```
Opens at: http://localhost:8000

### 6. Open the dashboard
Visit: http://localhost:8000

## Deployment (Render)

1. Push this repo to GitHub
2. Create a new **Web Service** on Render pointing to this repo
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `python api/main.py`
5. Add environment variables:
   - `RENDER_DB_INTERNAL_URL` (from your Render PostgreSQL)
   - `BINANCE_API_KEY`
   - `BINANCE_SECRET_KEY`
   - `ENVIRONMENT=production`
6. **Keep-alive**: Set up UptimeRobot (free) to ping `https://your-app.onrender.com/health` every 14 minutes

## How the Self-Improving Loop Works

```
Every 15 minutes:
  → Fetch new BTC candles from Binance
  → Run prediction with active LightGBM model
  → Check open trades for stop-loss / exit signals
  → Open new trade if confidence > 60%

Every 50 closed trades (per model):
  → Retrain LightGBM using warm-start (init_model=existing)
  → Model builds on what it already learned
  → New version saved if it improves
```

## The Three AI Models

| Model | Timeframe | Predicts | Checks |
|-------|-----------|----------|--------|
| AI-1  | 1-Hour candles | 5h ahead | Every 15 min |
| AI-2  | 8-Hour candles | 40h ahead | Every 15 min |
| AI-3  | 1-Day candles  | 5 days ahead | Every 15 min |

## Risk Rules (Always Active)

- Max 2% of portfolio per trade
- ATR-based stop-loss (1.5 × ATR below entry)
- Max 1 open trade per model
- Circuit breaker: model pauses if -5% daily loss
- Only trade if model confidence > 60%
