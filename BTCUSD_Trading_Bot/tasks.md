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

### Task 3: Run fetcher + train on Render deploy 🟡
- Bootstrap in main.py handles: drop tables → create fresh → fetch 15m → train models
- Will execute automatically when code is pushed to Render
- All progress visible in Render logs

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

### Task 7: Train models and review results 🟡
- Will run automatically during Render bootstrap
- Review F1 score in Render logs (target ≥ 47%)
- Review MAE in Render logs
- If unsatisfactory: adjust hyperparameters, redeploy
- Checkpoint: review results before enabling Phase 2 trading

---

## Phase 2: Live Bot (Automated)

### Task 8: Build combined predictor ✅
- models/predictor.py rewritten
- Loads direction + range_high + range_low models
- Returns: {direction, confidence, predicted_high, predicted_low, current_price}

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

### Task 11: Update scheduler to 15-min heartbeat ✅
- scheduler/job_runner.py rewritten
- Every 15 min: fetch → predict → log
- Trade execution will be added after Phase 2 tasks

### Task 12: Update retraining loop for both models ⬜
- Rewrite learning/retrain_loop.py
- Count closed trades, trigger at 100
- Retrain direction model (warm-start)
- Retrain range model (warm-start)
- Deploy new version only if equal or better score

### Task 13: Update database schema ✅
- data/database.py rewritten with new schema
- trades table: no stop_loss, has predicted_high/low
- portfolio table: USDT balance + BTC holdings
- predictions_log table: for accuracy tracking
- Bootstrap drops old tables and creates fresh on deploy

### Task 14: Update API + Dashboard ⬜
- api/routes.py updated for new model structure
- Dashboard UI still needs updating (Phase 2)
- Show predictions (direction + predicted high/low)
- Show portfolio state

### Task 15: Clean up old code ⬜
- Delete trading/risk_manager.py
- Remove old trainer.py (replaced by direction_trainer.py + range_trainer.py)
- Update README.md with new setup instructions

---

## Deployment Checklist (Render)

1. Push code to GitHub
2. Set Render env vars:
   - `ENVIRONMENT=production`
   - `RENDER_DB_INTERNAL_URL=postgresql://sanyam:RJrluq6tevyPaUqlFcIsAXpfbRnqJtTO@dpg-d9nhnjrncjis73a7dkeg-a/trading_bot_db_z23m`
   - (internal URL — no region prefix, uses Render internal network)
3. Build command: `pip install -r requirements.txt`
4. Start command: `python api/main.py`
5. Watch logs for bootstrap progress (fetch + train)
6. Once bootstrap completes, bot is live with 15-min heartbeat

---

## Execution Notes

- Phase 1 runs automatically on deploy (fetch data + train)
- Review training results in Render logs
- Phase 2 (actual trading) requires Tasks 9, 10, 12 to be implemented
- Code is clean, readable, well-commented — no vibe coding
