"""
decision/engine.py — Combines both model outputs into trade actions.

Logic:
  - BUY: direction=BUY + predicted_high is meaningfully above current price → buy $100-200 BTC
  - SELL: direction=SELL + bot holds BTC → sell all holdings
  - HOLD: do nothing
  - Exit by target: current_price >= predicted_high → sell (take profit)
  - Exit by range breach: current_price <= predicted_low → sell (prediction was wrong)

No stop-loss. No circuit breaker. AI learns freely.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from trading.portfolio_manager import (
    get_portfolio, buy_btc, sell_all_btc, record_trade
)
from data.database import log_event
from config import TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX, MIN_CONFIDENCE


MODEL_ID = "ai_15m"

# Minimum expected profit (as fraction of current price) to trigger a BUY
# If predicted_high is less than 0.1% above current price, not worth trading
MIN_EXPECTED_PROFIT_PCT = 0.001  # 0.1%


def make_decision(prediction: dict) -> dict:
    """
    Take a prediction from both models and decide what to do.

    Args:
        prediction: dict from predictor.predict() with:
            direction, confidence, predicted_high, predicted_low, current_price

    Returns:
        dict describing the action taken:
        {
            "action": "BUY" | "SELL" | "HOLD" | "EXIT_TARGET" | "EXIT_RANGE",
            "reason": str,
            "trade": dict or None
        }
    """
    if prediction is None:
        return {"action": "HOLD", "reason": "No prediction available", "trade": None}

    direction = prediction["direction"]
    confidence = prediction["confidence"]
    predicted_high = prediction["predicted_high"]
    predicted_low = prediction["predicted_low"]
    current_price = prediction["current_price"]

    portfolio = get_portfolio()
    btc_held = portfolio["btc_quantity"]
    has_btc = btc_held > 0

    # ── Check exit conditions for existing positions ──────────────────────────

    if has_btc:
        # Exit: current price reached predicted high (take profit)
        if current_price >= predicted_high:
            trade = sell_all_btc(current_price)
            if trade:
                record_trade(trade, prediction)
                return {
                    "action": "EXIT_TARGET",
                    "reason": f"Price ${current_price:,.0f} hit predicted high ${predicted_high:,.0f}. Selling.",
                    "trade": trade,
                }

        # Exit: current price dropped below predicted low (prediction was wrong)
        if current_price <= predicted_low:
            trade = sell_all_btc(current_price)
            if trade:
                record_trade(trade, prediction)
                return {
                    "action": "EXIT_RANGE",
                    "reason": f"Price ${current_price:,.0f} breached predicted low ${predicted_low:,.0f}. Cutting.",
                    "trade": trade,
                }

    # ── Direction-based decisions ─────────────────────────────────────────────

    # SELL signal: sell holdings if we have BTC
    if direction == "SELL" and has_btc:
        trade = sell_all_btc(current_price)
        if trade:
            record_trade(trade, prediction)
            return {
                "action": "SELL",
                "reason": f"Direction=SELL (conf={confidence:.1%}). Selling all BTC.",
                "trade": trade,
            }

    # BUY signal: buy if conditions are met
    if direction == "BUY" and confidence >= MIN_CONFIDENCE:
        # Check if predicted high is meaningfully above current price
        expected_profit_pct = (predicted_high - current_price) / current_price

        if expected_profit_pct >= MIN_EXPECTED_PROFIT_PCT:
            # Pick random amount between $100-$200
            amount_usdt = random.uniform(TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX)
            amount_usdt = round(amount_usdt, 2)

            trade = buy_btc(amount_usdt, current_price)
            if trade:
                record_trade(trade, prediction)
                return {
                    "action": "BUY",
                    "reason": (f"Direction=BUY (conf={confidence:.1%}), "
                               f"expected +{expected_profit_pct:.2%} to ${predicted_high:,.0f}. "
                               f"Buying ${amount_usdt:.0f} of BTC."),
                    "trade": trade,
                }
        else:
            return {
                "action": "HOLD",
                "reason": (f"Direction=BUY but predicted high ${predicted_high:,.0f} "
                           f"is only +{expected_profit_pct:.3%} above current ${current_price:,.0f}. "
                           f"Not worth it."),
                "trade": None,
            }

    # HOLD or low confidence
    reason = f"Direction={direction} (conf={confidence:.1%})"
    if direction == "BUY" and confidence < MIN_CONFIDENCE:
        reason += f" — below {MIN_CONFIDENCE:.0%} threshold"
    if direction == "SELL" and not has_btc:
        reason += " — but no BTC to sell"

    return {"action": "HOLD", "reason": reason, "trade": None}
