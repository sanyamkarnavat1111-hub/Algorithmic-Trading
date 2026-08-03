# Trading Bot Revamp — Task List

> Status: ⬜ = Not Started | 🟡 = In Progress | ✅ = Done | ❌ = Blocked

---

## Phase 1: Data + Training (Manual, Iterative)

### Task 1: Update config.py for 15-minute timeframe ✅
- Change TIMEFRAMES to "15m"
- Set HEARTBEAT_MINUTES = 15
- Remove stop-loss config (STOP_LOSS_ATR_MULT, DAILY_LOSS_LIMIT, circuit breaker)
- Remove PAPER_STARTING_CAPITAL (replace with large balance)
- Add TRADE_AMOUNT_MIN = 100, TRADE_AMOUNT_MAX = 200
- Keep LABEL_LOOKAHEAD = 5
- Add LightGBM params for regression model
- Adjust BUY/SELL thresholds for 15-min (±0.3%)

### Task 2: Update binance_fetcher.py for 15-min candles ✅
- Change interval to "15m"
- Update entry point to sync "15m"
- Keep incremental sync, pagination, retry logic
- Verify Binance API supports 15m interval

### Task 3: Run fetcher to populate database ⬜
- Execute fetcher script
- Verify data with check_db.py
- Confirm we have enough rows for training (need 250+ after indicator warmup)

### Task 4: Update feature pipeline for dual labels ✅
- Keep all existing indicators (they work on any timeframe)
- Add HIGH label: max(high) across next 5 candles
- Add LOW label: min(low) across next 5 candles
- Keep direction label: BUY/HOLD/SELL based on 5-candle close-to-close return
- Adjust threshold to ±0.3% for 15-min timeframe
- Save separate scalers for direction model and range model

### Task 5: Build Direction Model trainer ✅
- New file: models/direction_trainer.py
- LightGBM multiclass classification (BUY/HOLD/SELL)
- Time-series K-fold cross validation (5 folds, walk-forward, no future leakage)
- Class-weight balancing (HOLD is majority)
- Early stopping on validation multi_logloss
- Print: F1 weighted (target ≥ 47%), F1 macro, confusion matrix, classification report
- Print: feature importance top 10
- Save to model registry with version
- Support warm-start for retraining

### Task 6: Build Range Model trainer ✅
- New file: models/range_trainer.py
- Two LightGBM regressors: one for predicted HIGH, one for predicted LOW
- Regression objective (MAE or Huber loss)
- Time-series K-fold cross validation (5 folds, walk-forward)
- Early stopping on validation loss
- Print: MAE, RMSE, R² for both high and low
- Print: average error in dollar terms
- Save both models to registry
- Support warm-start for retraining

### Task 7: Train models and review results ⬜
- Run: python models/direction_trainer.py → review F1 score
- Run: python models/range_trainer.py → review MAE
- If unsatisfactory: adjust hyperparameters, retrain
- Iterate until both models meet targets
- Checkpoint: STOP HERE, review with user before Phase 2

---

## Phase 2: Live Bot (Automated)

### Task 8: Build combined predictor ⬜
- Rewrite models/predictor.py
- Load both active models (direction + range HIGH + range LOW)
- Return: {direction, confidence, predicted_high, predicted_low, current_price}

### Task 9: Build portfolio manager ⬜
- New file: trading/portfolio_manager.py
- Track: USDT balance, BTC quantity, average buy price
- buy_btc(amount_usdt, price) and sell_btc(quantity, price)
- Cannot sell more BTC than held (no shorting)
- Multiple open positions allowed
- Store transactions in trades table
- No stop-loss, no circuit breaker

### Task 10: Build decision engine ⬜
- New file: decision/engine.py
- Combines both model outputs into trade actions
- BUY: direction=BUY + predicted_high meaningfully above current price → buy $100-200 BTC
- SELL: direction=SELL + bot holds BTC → sell
- HOLD: do nothing
- Exit if current_price ≥ predicted_high (target hit)
- Exit if current_price ≤ predicted_low (prediction wrong, cut)
- Log reasoning for each decision

### Task 11: Update scheduler to 15-min heartbeat ⬜
- Rewrite scheduler/job_runner.py
- Every 15 min: fetch → predict → decide → check positions
- Trigger retraining after 100 closed trades

### Task 12: Update retraining loop for both models ⬜
- Rewrite learning/retrain_loop.py
- Count closed trades, trigger at 100
- Retrain direction model (warm-start)
- Retrain range model (warm-start)
- Deploy new version only if equal or better score

### Task 13: Update database schema ⬜
- Remove stop_loss column from trades
- Add predicted_high, predicted_low to trades
- Add portfolio state table
- Add predictions log table (for accuracy tracking)

### Task 14: Update API + Dashboard ⬜
- Show predictions (direction + predicted high/low) on dashboard
- Show portfolio state (USDT + BTC + total value)
- Show prediction accuracy (predicted vs actual)
- Remove stop-loss references from UI

### Task 15: Clean up old code ⬜
- Delete trading/risk_manager.py
- Remove all stop-loss and circuit breaker logic
- Remove sqlalchemy from requirements.txt
- Update README.md with new setup instructions
- Update .gitignore if needed

---

## Execution Notes

- Phase 1 is done manually and iteratively (train → review → adjust → retrain)
- Phase 2 only starts after Phase 1 models are satisfactory
- Each task will be executed one at a time with user review
- Code will be clean, readable, well-commented — no vibe coding
