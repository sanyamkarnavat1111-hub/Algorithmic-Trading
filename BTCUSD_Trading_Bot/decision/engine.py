"""
decision/engine.py — Combines both model outputs into trade actions.

NO THRESHOLDS. Whatever the direction model's highest vote is (BUY/SELL/HOLD),
that's what happens. The AI decides freely. It learns from mistakes via retraining.

Logic:
  - BUY: direction model says BUY → buy $100-200 of BTC
  - SELL: direction model says SELL + bot holds BTC → sell all
  - HOLD: direction model says HOLD → do nothing
  - Exit by target: current price >= predicted high → sell
  - Exit by range breach: current price <= predicted low → sell
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from trading.portfolio_manager import (
    get_portfolio, buy_btc, sell_all_btc, record_trade, record_hold
)
from data.database import log_event
from config import TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX


MODEL_ID = "ai_15m"


def make_decision(prediction: dict) -> dict:
    """
    Take a prediction and act on it. No thresholds — majority vote wins.

    Returns:
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
        # Exit: current price reached or exceeded predicted high (take profit)
        if current_price >= predicted_high:
            trade = sell_all_btc(current_price)
            if trade:
                record_trade(trade, prediction)
                return {
                    "action": "EXIT_TARGET",
                    "reason": f"Price ${current_price:,.0f} hit predicted high ${predicted_high:,.0f}. Taking profit.",
                    "trade": trade,
                }

        # Exit: current price dropped to or below predicted low (AI was wrong)
        if current_price <= predicted_low:
            trade = sell_all_btc(current_price)
            if trade:
                record_trade(trade, prediction)
                return {
                    "action": "EXIT_RANGE",
                    "reason": f"Price ${current_price:,.0f} hit predicted low ${predicted_low:,.0f}. Exiting.",
                    "trade": trade,
                }

    # ── Act on direction model's majority vote (NO threshold) ─────────────────

    if direction == "SELL" and has_btc:
        trade = sell_all_btc(current_price)
        if trade:
            record_trade(trade, prediction)
            return {
                "action": "SELL",
                "reason": f"AI says SELL ({confidence:.1%}). Sold all BTC.",
                "trade": trade,
            }

    if direction == "SELL" and not has_btc:
        record_hold(prediction, "AI says SELL but no BTC held")
        return {
            "action": "HOLD",
            "reason": "AI says SELL but no BTC to sell.",
            "trade": None,
        }

    if direction == "BUY":
        amount_usdt = random.uniform(TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX)
        amount_usdt = round(amount_usdt, 2)

        trade = buy_btc(amount_usdt, current_price)
        if trade:
            record_trade(trade, prediction)
            return {
                "action": "BUY",
                "reason": f"AI says BUY ({confidence:.1%}). Bought ${amount_usdt:.0f} of BTC. Target: ${predicted_high:,.0f}",
                "trade": trade,
            }

    # HOLD
    record_hold(prediction, f"AI says HOLD ({confidence:.1%})")
    return {
        "action": "HOLD",
        "reason": f"AI says HOLD ({confidence:.1%}). Waiting.",
        "trade": None,
    }
