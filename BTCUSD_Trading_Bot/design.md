# Trading Bot Revamp — Final Design

## The Big Picture

A BTC/USDT spot trading bot that uses two AI models working together. One predicts direction (BUY/SELL/HOLD), the other predicts the expected HIGH and LOW prices for the next 75 minutes. The bot buys and sells BTC simply — like a person would on Binance spot market. No futures, no shorting, no options, no contracts. Just buy BTC when AI says it's going up, sell BTC when AI says it's going down.

---

## How It Works End-to-End

### Step 1: Data

Every 15 minutes, the bot fetches the latest 15-minute BTC/USDT candle from Binance and stores it in PostgreSQL. Historical data from 2017 is also fetched on first run for training.

### Step 2: Two AI Models

**Direction Model (Classification — LightGBM)**
Looks at current market conditions (technical indicators, time features) and predicts: will price go UP, DOWN, or SIDEWAYS in the next 5 candles (75 minutes)?
Output: BUY / SELL / HOLD + confidence percentage.

**Range Model (Regression — LightGBM)**
Looks at the same market conditions and predicts two numbers:
- Predicted HIGH: the highest price BTC will reach in the next 75 minutes
- Predicted LOW: the lowest price BTC will drop to in the next 75 minutes

No close prediction. Just high and low — that's enough to make decisions.

### Step 3: Decision Making

Every 15 minutes after both models run:

- **Direction says BUY + Range says high is meaningfully above current price** → Buy $100-$200 worth of BTC at current market price. The bot can buy multiple times (stack positions).

- **Direction says SELL + bot holds BTC** → Sell some or all BTC holdings at current market price.

- **Direction says HOLD** → Do nothing. Keep existing positions.

- **If bot has no BTC and direction says SELL** → Do nothing. Can't sell what you don't have.

Exit logic is simple:
- Model says SELL → sell what you hold
- Current price reaches the predicted HIGH → sell (take profit)
- Current price drops below predicted LOW → sell (the prediction was wrong, cut losses)
- After 5 candles pass, re-evaluate with fresh predictions

No stop-loss. No circuit breaker. Let the AI trade freely, learn from mistakes, and improve over time.

### Step 4: Portfolio Tracking

- Starting USDT balance: very large (effectively unlimited, say $10,000,000)
- Each trade uses $100-$200 worth of BTC (model decides within this range)
- Bot tracks: USDT balance, BTC holdings (quantity), average buy price, total portfolio value
- Multiple open positions allowed (bot can buy multiple times)
- Simple accounting: buy reduces USDT and increases BTC, sell does the opposite

### Step 5: Retraining Loop

After every 100 closed trades:
- Retrain the Direction Model on all historical data (warm-start)
- Retrain the Range Model on all historical data (warm-start)
- Both use proper ML practices: scaling, normalization, K-fold cross validation, walk-forward splits
- New model only goes live if it scores equal or better than current (F1 ≥ 47% target for direction, MAE improvement for range)
- Both models versioned independently

### Step 6: Dashboard

Shows: current price, AI predictions (direction + predicted high/low), portfolio state (USDT + BTC + total value), recent trades with P&L, model accuracy stats.

---

## ML Training Best Practices (Applied to Both Models)

- StandardScaler normalization on all features
- Time-series aware K-fold cross validation (no future data leakage)
- Walk-forward validation (train on past, test on future)
- Class-weight balancing for direction model (HOLD is majority class)
- Feature importance analysis after training
- Early stopping on validation loss
- Hyperparameter tuning (num_leaves, learning_rate, etc.)
- Warm-start for online learning (model builds on previous version)
- Separate scalers saved for each model

---

## Execution Phases

**Phase 1: Data + Training (Manual, Iterative)**
1. Update fetcher script to pull 15-minute candles
2. Run fetcher to populate DB with historical 15-min data
3. Train Direction Model — review F1 score, confusion matrix, feature importance
4. Train Range Model — review MAE, prediction accuracy
5. If scores unsatisfactory → adjust parameters → retrain
6. Iterate until we're happy with both models

**Phase 2: Live Bot (Automated)**
7. Wire up the decision engine (combine both model outputs)
8. Implement portfolio manager (buy/sell BTC, track balances)
9. Set up 15-min scheduler
10. Update API + dashboard to show new predictions and portfolio
11. Set up retraining loop for both models
12. Deploy to Render

---

## What's Removed

- Stop-loss (gone entirely)
- Circuit breaker (gone)
- Shorting / futures / options (not applicable — spot only)
- Multiple timeframes (just 15-min)
- ATR-based exits
- 60-min heartbeat (now 15-min)
- Arbitrary $1M starting with 2% sizing (now unlimited with $100-200 per trade)

---

## What's New

- Range prediction model (high/low)
- Multiple open positions
- AI-driven exits (predicted high/low determines exit)
- 15-min candles
- K-fold cross validation in training
- Proper iterative training workflow (train → review → adjust → retrain)
- Portfolio with explicit BTC holdings tracking
- Prediction accuracy visible on dashboard
