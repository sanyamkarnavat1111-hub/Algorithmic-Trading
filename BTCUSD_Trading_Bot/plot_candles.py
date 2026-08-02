"""
plot_candles.py — Plots the last N candles for all 3 timeframes side-by-side.
Run this script to visually validate the pulled data looks correct.

Usage:
    python plot_candles.py              # shows last 200 candles per timeframe
    python plot_candles.py --limit 500  # shows last 500 candles per timeframe
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from data.binance_fetcher import load_candles

# ── Parse CLI args ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Plot BTC/USDT candlestick charts from database.")
parser.add_argument("--limit", type=int, default=200, help="Number of candles to plot per timeframe (default: 200)")
args = parser.parse_args()

LIMIT = args.limit
TIMEFRAMES = ["1h", "8h", "1d"]

print(f"[Plot] Loading last {LIMIT} candles for each timeframe...")

# ── Load and prepare data ─────────────────────────────────────────────────────

datasets = {}
for tf in TIMEFRAMES:
    df = load_candles(tf, limit=LIMIT)
    if df.empty:
        print(f"[Plot] WARNING: No data for {tf}. Skipping.")
        continue

    # mplfinance requires DatetimeIndex with OHLCV columns
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time")
    df = df.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    datasets[tf] = df
    print(f"[Plot] {tf}: {len(df)} candles | {df.index.min()} -> {df.index.max()} | "
          f"Close range: ${df['Close'].min():,.0f} - ${df['Close'].max():,.0f}")

if not datasets:
    print("[Plot] No data found in database. Run binance_fetcher.py first.")
    sys.exit(1)

# ── Plot each timeframe as a separate window ──────────────────────────────────

style = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    gridstyle=":",
    y_on_right=True,
)

for tf, df in datasets.items():
    print(f"\n[Plot] Opening chart for {tf} timeframe...")
    mpf.plot(
        df,
        type="candle",
        style=style,
        title=f"\nBTC/USDT — {tf.upper()} Candles (last {len(df)} candles)",
        ylabel="Price (USDT)",
        ylabel_lower="Volume",
        volume=True,
        figsize=(16, 8),
        show_nontrading=False,
    )

print("\n[Plot] All charts displayed. Close the chart windows to exit.")
