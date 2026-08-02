"""
features/time_features.py — Cyclic time-based features.

Encodes hour, day-of-week, and month as sin/cos pairs so the model
understands the circular/cyclic nature of time.

Example:
  Hour 23 and Hour 0 are close to each other — sin/cos encoding
  captures this correctly, unlike raw integers.

This helps the model learn:
  - Which trading session is active (London, NY, Asian)
  - Monday gap-fills and Friday profit-taking patterns
  - Seasonal macro patterns (e.g., Fed rate decision months)
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time features to a DataFrame that has an 'open_time' column.

    Input:  DataFrame with 'open_time' column (UTC datetime)
    Output: Same DataFrame with 6 new time feature columns

    The sin/cos encoding maps circular values to a unit circle:
      f(x) = sin(2π × x / max_x),  cos(2π × x / max_x)

    This ensures e.g. hour=23 and hour=0 are "close" to each other.
    """
    df = df.copy()

    # Extract raw time components
    hour        = df["open_time"].dt.hour         # 0–23
    day_of_week = df["open_time"].dt.dayofweek    # 0=Monday, 6=Sunday
    month       = df["open_time"].dt.month        # 1–12

    # ── Hour of Day (trading session proxy) ───────────────────────────────────
    # London session: 8–16 UTC | NY session: 13–21 UTC | Asian: 0–8 UTC
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # ── Day of Week ───────────────────────────────────────────────────────────
    # Captures: Monday gaps, mid-week momentum, Friday risk-off behaviour
    df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # ── Month of Year ─────────────────────────────────────────────────────────
    # Captures: January effect, summer doldrums, Q4 rallies,
    # FOMC meeting months (Jan, Mar, May, Jun, Jul, Sep, Nov, Dec)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df
