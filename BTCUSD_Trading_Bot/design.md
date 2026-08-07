# Trading Bot — Final Trading Logic (Confirmed)

## Entry Rule
- Only BUY positions (no shorting, no futures)
- AI Direction Model says BUY → open a BUY position ($100-200)
- AI Direction Model says SELL or HOLD → do NOT open anything
- Max 5 positions open at same time
- 1 new position per candle (staggered)

## Exit Rules (2 conditions only)

**Take Profit: +10%**
- If current price is +10% above entry price → SELL immediately

**Stop Loss: -5%**
- If actual current price is already -5% below entry → SELL immediately
- OR if AI (Direction + Range Model) predicts price will go DOWN and the predicted low is -5% or more below entry → SELL immediately (AI-assisted early exit)

## How Range Model Helps

The Range Model predicts how low price CAN go in the next 75 min. If:
- Direction Model says SELL (price going down)
- AND Range Model's predicted low is ≥ 5% below a position's entry price
- → That position gets closed early (before actual -5% hit)

This is the AI helping you exit BEFORE the actual loss happens. Smart early warning.

## The 15-Minute Cycle

```
Every 15 minutes:
  1. Fetch latest candle
  2. Get current price
  3. Run AI prediction (direction + predicted high/low)
  4. For each OPEN position:
     a. If current_price >= entry_price × 1.10 → CLOSE (🎯 +10% profit)
     b. If current_price <= entry_price × 0.95 → CLOSE (❌ -5% actual loss)
     c. If direction=SELL AND predicted_low <= entry_price × 0.95 → CLOSE (⚠️ AI predicts -5% coming)
     d. Otherwise → HOLD
  5. If open_positions < 5 AND direction = BUY:
     → OPEN new BUY position at current price
```

## Summary
- Only BUY. Never short.
- SELL signal = don't open new + check if existing positions need early exit
- +10% take profit, -5% stop loss (1:2 risk reward)
- Range Model gives early warning for exits
- 5 parallel positions, staggered 1 per candle
