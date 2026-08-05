"""
decision/engine.py — Combines both model outputs into trade actions.

NO THRESHOLDS. Whatever the direction model's highest vote is (BUY/SELL/HOLD),
that's what happens. The AI decides freely. It learns from mistakes via retraining.

Trade sizing:
  - BUY: spend $100-200 USDT to buy BTC
  - SELL: sell $100-200 worth of BTC

Logic:
  - BUY: direction model says BUY → buy $100-200 of BTC
  - SELL: direction model says SELL → sell $100-200 worth of BTC
  - HOLD: direction model says HOLD → do nothing
  - Exit by target: current price >= predicted high → sell $100-200 worth
  - Exit by range breach: current price <= predicted low → sell $100-200 worth
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from trading.portfolio_manager import (
    get_portfolio, buy_btc, sell_btc, record_trade, record_hold
)
from data.database import log_event
from config import TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX


MODEL_ID = "ai_15m"


def _get_sell_quantity(current_price: float) -> float:
    """Calculate how much BTC to sell ($100-200 worth)."""
    amount_usdt = random.uniform(TRADE_AMOUNT_MIN, TRADE_AMOUNT_MAX)
    btc_quantity = amount_usdt / current_price
    return round(btc_quantity, 8)


def make_decision(prediction: dict) -> dict:
    """
    Take a prediction and act on it. No thresholds — majority vote wins.

    Trade sizing:
      BUY → spend $100-200 USDT to get BTC
      SELL → sell $100-200 worth of BTC from holdings

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
            sell_qty = _get_sell_quantity(current_price)
            sell_qty = min(sell_qty, btc_held)  # can't sell more than we have
            if sell_qty > 0:
                trade = sell_btc(sell_qty, current_price)
                if trade:
                    record_trade(trade, prediction)
                    return {
                        "action": "EXIT_TARGET",
                        "reason": f"Price ${current_price:,.0f} hit predicted high ${predicted_high:,.0f}. Sold ${trade['amount_usdt']:.0f} worth.",
                        "trade": trade,
                    }

        # Exit: current price dropped to or below predicted low (AI was wrong)
        if current_price <= predicted_low:
            sell_qty = _get_sell_quantity(current_price)
            sell_qty = min(sell_qty, btc_held)
            if sell_qty > 0:
                trade = sell_btc(sell_qty, current_price)
                if trade:
                    record_trade(trade, prediction)
                    return {
                        "action": "EXIT_RANGE",
                        "reason": f"Price ${current_price:,.0f} hit predicted low ${predicted_low:,.0f}. Sold ${trade['amount_usdt']:.0f} worth.",
                        "trade": trade,
                    }

    # ── Act on direction model's majority vote ────────────────────────────────

    if direction == "SELL":
        if has_btc:
            sell_qty = _get_sell_quantity(current_price)
            sell_qty = min(sell_qty, btc_held)  # can't sell more than we have
            if sell_qty > 0:
                trade = sell_btc(sell_qty, current_price)
                if trade:
                    record_trade(trade, prediction)
                    return {
                        "action": "SELL",
                        "reason": f"AI says SELL ({confidence:.1%}). Sold ${trade['amount_usdt']:.0f} worth of BTC.",
                        "trade": trade,
                    }
        # If somehow btc_held is 0 or too small
        record_hold(prediction, "AI says SELL but insufficient BTC")
        return {"action": "HOLD", "reason": "AI says SELL but not enough BTC to sell.", "trade": None}

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
