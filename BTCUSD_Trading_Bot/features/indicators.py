"""
features/indicators.py — Compute all technical indicators using pandas-ta.

Uses dynamic column name discovery so this is resilient to pandas-ta version differences
that change column naming (e.g. 'BBU_20_2.0' vs 'BBU_20_2').
"""

import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a raw OHLCV DataFrame.

    Input:  DataFrame with columns [open_time, open, high, low, close, volume]
    Output: Same DataFrame with indicator columns added. NaN rows dropped.
    """
    df = df.copy()

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── Momentum ──────────────────────────────────────────────────────────────

    df["rsi_14"]  = ta.rsi(close, length=14)

    stoch = ta.stoch(high, low, close, k=14, d=3)
    if stoch is not None and not stoch.empty:
        # Grab first two columns regardless of exact name (k, d)
        cols = list(stoch.columns)
        df["stoch_k"] = stoch[cols[0]]
        df["stoch_d"] = stoch[cols[1]]

    # ── Trend ─────────────────────────────────────────────────────────────────

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        cols = list(macd_df.columns)
        # pandas-ta MACD returns: [MACD, MACDh, MACDs] — find by prefix
        for col in cols:
            col_upper = col.upper()
            if col_upper.startswith("MACD_"):
                df["macd"] = macd_df[col]
            elif col_upper.startswith("MACDS_"):
                df["macd_signal"] = macd_df[col]
            elif col_upper.startswith("MACDH_"):
                df["macd_hist"] = macd_df[col]

    df["ema_9"]   = ta.ema(close, length=9)
    df["ema_21"]  = ta.ema(close, length=21)
    df["ema_50"]  = ta.ema(close, length=50)
    df["ema_200"] = ta.ema(close, length=200)

    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None and not adx_df.empty:
        # First column is ADX value
        df["adx"] = adx_df.iloc[:, 0]

    # ── Volatility ────────────────────────────────────────────────────────────

    bbands = ta.bbands(close, length=20, std=2)
    if bbands is not None and not bbands.empty:
        # Columns are always in order: [BBL, BBM, BBU, BBB, BBP] (lower, mid, upper, bandwidth, percent)
        for col in bbands.columns:
            col_upper = col.upper()
            if col_upper.startswith("BBL_"):
                df["bb_lower"] = bbands[col]
            elif col_upper.startswith("BBM_"):
                df["bb_mid"]   = bbands[col]
            elif col_upper.startswith("BBU_"):
                df["bb_upper"] = bbands[col]

        # Compute derived features
        if "bb_upper" in df and "bb_lower" in df and "bb_mid" in df:
            df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)
            df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    df["atr_14"] = ta.atr(high, low, close, length=14)

    # ── Volume ────────────────────────────────────────────────────────────────

    df["obv"] = ta.obv(close, volume)

    # VWAP — pandas-ta requires a DatetimeIndex, so we temporarily set one
    try:
        # Build a temp df with DatetimeIndex for ta.vwap()
        if "open_time" in df.columns:
            tmp = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["open_time"], utc=True)))
        else:
            # open_time is already the index
            tmp = df.copy()
            if not isinstance(tmp.index, pd.DatetimeIndex):
                raise ValueError("No DatetimeIndex available for VWAP")
        vwap_series = ta.vwap(tmp["high"], tmp["low"], tmp["close"], tmp["volume"])
        if vwap_series is not None:
            df["vwap"] = vwap_series.values
        else:
            raise ValueError("ta.vwap returned None")
    except Exception as e:
        # Fallback: rolling VWAP using typical price (always works)
        tp = (high + low + close) / 3.0
        df["vwap"] = (tp * volume).rolling(14, min_periods=1).sum() / (volume.rolling(14, min_periods=1).sum() + 1e-9)


    # ── Price-Action Derived Features ─────────────────────────────────────────

    df["candle_body"]         = close - df["open"]
    df["candle_range"]        = high - low
    df["body_to_range_ratio"] = df["candle_body"] / (df["candle_range"] + 1e-9)

    # Distance from key EMAs (normalised)
    df["close_to_ema21"]  = (close - df["ema_21"])  / (df["ema_21"]  + 1e-9)
    df["close_to_ema200"] = (close - df["ema_200"]) / (df["ema_200"] + 1e-9)

    # ── Drop NaN rows (from indicator warm-up) ────────────────────────────────
    df = df.dropna().reset_index(drop=True)

    return df
