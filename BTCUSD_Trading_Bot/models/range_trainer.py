"""
models/range_trainer.py — Train the Range Model (predict HIGH and LOW prices).

Uses two separate LightGBM regressors:
  - HIGH model: predicts the maximum price in the next 5 candles (75 min)
  - LOW model: predicts the minimum price in the next 5 candles (75 min)

Training practices:
  - Time-series K-fold cross validation (walk-forward, no future leakage)
  - Early stopping on validation MAE
  - StandardScaler normalization
  - Warm-start support for online learning

Usage:
    python models/range_trainer.py              # fresh train
    python models/range_trainer.py --warm       # warm-start retrain
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from data.binance_fetcher import load_candles
from features.pipeline import build_features, generate_range_labels, get_feature_columns
from models.registry import save_model, load_active_model
from data.database import log_event
from config import RANGE_LGBM_PARAMS, TIMEFRAMES


TIMEFRAME = "15m"
SCALER_ID = "range_15m"


def train(warm_start: bool = False) -> dict:
    """
    Train the Range Model (HIGH and LOW predictors).

    Process:
      1. Load all 15-min candles from DB
      2. Build features + generate range labels (actual high/low of next 5 candles)
      3. Time-series K-fold cross validation (5 folds)
      4. Final train on 70% / validate on 15% / test on 15%
      5. Report MAE, RMSE, R² for both HIGH and LOW models
      6. Save both models

    Returns dict with metrics.
    """
    model_id = TIMEFRAMES[TIMEFRAME]["model_id"]
    print(f"\n{'='*60}")
    print(f"[Range Trainer] {'Warm-start' if warm_start else 'Fresh'} training")
    print(f"{'='*60}")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    raw_df = load_candles(TIMEFRAME, limit=None)
    if raw_df.empty:
        raise ValueError("No candle data. Run binance_fetcher.py first.")
    print(f"[Data] Loaded {len(raw_df)} raw candles.")

    # ── Step 2: Build features ────────────────────────────────────────────────
    feature_df = build_features(raw_df, fit_scaler=True, scaler_id=SCALER_ID)
    print(f"[Features] {len(feature_df)} rows after indicator warmup.")

    # ── Step 3: Generate range labels ─────────────────────────────────────────
    label_df = raw_df.iloc[-len(feature_df):].reset_index(drop=True).copy()
    label_df = generate_range_labels(label_df)

    # Trim feature_df to match
    feature_df = feature_df.iloc[:len(label_df)].reset_index(drop=True)

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in feature_df.columns]

    X = feature_df[available].values
    y_high = label_df["range_high_label"].values.astype(float)
    y_low = label_df["range_low_label"].values.astype(float)

    # Also get current close for context in reporting
    current_close = label_df["close"].values.astype(float)

    print(f"[Data] Final dataset: {len(X)} rows × {len(available)} features")
    print(f"[Labels] High return range: {y_high.min()*100:+.2f}% to {y_high.max()*100:+.2f}%")
    print(f"[Labels] Low return range:  {y_low.min()*100:+.2f}% to {y_low.max()*100:+.2f}%")

    # ── Step 4: K-Fold Cross Validation ───────────────────────────────────────
    print(f"\n[CV] Running 5-fold time-series cross validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae_high = []
    cv_mae_low = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        yh_tr, yh_vl = y_high[train_idx], y_high[val_idx]
        yl_tr, yl_vl = y_low[train_idx], y_low[val_idx]

        # Train HIGH model
        h_train = lgb.Dataset(X_tr, label=yh_tr, feature_name=available)
        h_val = lgb.Dataset(X_vl, label=yh_vl, reference=h_train)
        h_model = lgb.train(
            params={**RANGE_LGBM_PARAMS, "verbosity": -1},
            train_set=h_train, num_boost_round=300,
            valid_sets=[h_val],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        # Train LOW model
        l_train = lgb.Dataset(X_tr, label=yl_tr, feature_name=available)
        l_val = lgb.Dataset(X_vl, label=yl_vl, reference=l_train)
        l_model = lgb.train(
            params={**RANGE_LGBM_PARAMS, "verbosity": -1},
            train_set=l_train, num_boost_round=300,
            valid_sets=[l_val],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        h_pred = h_model.predict(X_vl)
        l_pred = l_model.predict(X_vl)

        mae_h = mean_absolute_error(yh_vl, h_pred)
        mae_l = mean_absolute_error(yl_vl, l_pred)
        cv_mae_high.append(mae_h)
        cv_mae_low.append(mae_l)
        print(f"  Fold {fold}: HIGH MAE = {mae_h*100:.3f}% | LOW MAE = {mae_l*100:.3f}%")

    print(f"\n[CV] Mean HIGH MAE: {np.mean(cv_mae_high)*100:.3f}%")
    print(f"[CV] Mean LOW MAE:  {np.mean(cv_mae_low)*100:.3f}%")

    # ── Step 5: Final train/val/test split (70/15/15) ─────────────────────────
    print(f"\n[Final] Training on 70% / Val 15% / Test 15%...")
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    yh_train, yh_val, yh_test = y_high[:train_end], y_high[train_end:val_end], y_high[val_end:]
    yl_train, yl_val, yl_test = y_low[:train_end], y_low[train_end:val_end], y_low[val_end:]
    close_test = current_close[val_end:]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Adaptive complexity
    n_train = len(X_train)
    if n_train < 10000:
        num_leaves, min_child = 31, 20
    elif n_train < 50000:
        num_leaves, min_child = 63, 30
    else:
        num_leaves, min_child = 127, 50

    params = {
        **RANGE_LGBM_PARAMS,
        "verbosity": -1,
        "num_leaves": num_leaves,
        "min_child_samples": min_child,
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    # ── Train HIGH model ──────────────────────────────────────────────────────
    h_train_data = lgb.Dataset(X_train, label=yh_train, feature_name=available)
    h_val_data = lgb.Dataset(X_val, label=yh_val, reference=h_train_data)

    init_high = None
    if warm_start:
        init_high = load_active_model(TIMEFRAME, "range_high")
        if init_high:
            print(f"  Loaded existing HIGH model for warm-start.")

    high_model = lgb.train(
        params=params,
        train_set=h_train_data,
        num_boost_round=500,
        valid_sets=[h_val_data],
        callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)],
        init_model=init_high,
    )

    # ── Train LOW model ───────────────────────────────────────────────────────
    l_train_data = lgb.Dataset(X_train, label=yl_train, feature_name=available)
    l_val_data = lgb.Dataset(X_val, label=yl_val, reference=l_train_data)

    init_low = None
    if warm_start:
        init_low = load_active_model(TIMEFRAME, "range_low")
        if init_low:
            print(f"  Loaded existing LOW model for warm-start.")

    low_model = lgb.train(
        params=params,
        train_set=l_train_data,
        num_boost_round=500,
        valid_sets=[l_val_data],
        callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)],
        init_model=init_low,
    )

    # ── Step 6: Evaluate on test set ──────────────────────────────────────────
    h_pred = high_model.predict(X_test)
    l_pred = low_model.predict(X_test)

    # HIGH metrics
    mae_high = mean_absolute_error(yh_test, h_pred)
    rmse_high = np.sqrt(mean_squared_error(yh_test, h_pred))
    r2_high = r2_score(yh_test, h_pred)

    # LOW metrics
    mae_low = mean_absolute_error(yl_test, l_pred)
    rmse_low = np.sqrt(mean_squared_error(yl_test, l_pred))
    r2_low = r2_score(yl_test, l_pred)

    # Approximate dollar error relative to average price
    avg_price = np.mean(close_test)
    dollar_mae_high = mae_high * avg_price
    dollar_mae_low = mae_low * avg_price

    print(f"\n{'='*60}")
    print(f"[RESULTS] Range Model — Test Set")
    print(f"{'='*60}")
    print(f"\n  HIGH Prediction:")
    print(f"    MAE   : {mae_high*100:.3f}%  (approx ${dollar_mae_high:,.2f})")
    print(f"    RMSE  : {rmse_high*100:.3f}%")
    print(f"    R²    : {r2_high:.4f}")
    print(f"\n  LOW Prediction:")
    print(f"    MAE   : {mae_low*100:.3f}%  (approx ${dollar_mae_low:,.2f})")
    print(f"    RMSE  : {rmse_low*100:.3f}%")
    print(f"    R²    : {r2_low:.4f}")
    print(f"\n  CV Mean HIGH MAE: {np.mean(cv_mae_high)*100:.3f}%")
    print(f"  CV Mean LOW MAE:  {np.mean(cv_mae_low)*100:.3f}%")

    # Sample predictions vs actuals (reconstructed dollar values)
    print(f"\n  Sample predictions (last 5 in test set):")
    print(f"  {'Close':>10} | {'Pred High':>10} | {'Actual High':>11} | {'Pred Low':>10} | {'Actual Low':>10}")
    print(f"  {'-'*60}")
    for i in range(-5, 0):
        pred_h_usd = close_test[i] * (1 + h_pred[i])
        actual_h_usd = close_test[i] * (1 + yh_test[i])
        pred_l_usd = close_test[i] * (1 + l_pred[i])
        actual_l_usd = close_test[i] * (1 + yl_test[i])
        
        print(f"  ${close_test[i]:>9,.0f} | ${pred_h_usd:>9,.0f} | ${actual_h_usd:>10,.0f} | "
              f"${pred_l_usd:>9,.0f} | ${actual_l_usd:>9,.0f}")

    # Feature importance (from HIGH model)
    importance = high_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(available, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features by gain (HIGH model):")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat:<25} {imp:.1f}")

    # ── Step 7: Save models ───────────────────────────────────────────────────
    v_high = save_model(high_model, TIMEFRAME, "range_high",
                        accuracy=mae_high, train_rows=len(X_train))
    v_low = save_model(low_model, TIMEFRAME, "range_low",
                       accuracy=mae_low, train_rows=len(X_train))

    log_event("INFO",
              f"Range models trained: HIGH MAE=${mae_high:.2f}, LOW MAE=${mae_low:.2f}",
              model_id=model_id)

    print(f"\n[SAVED] HIGH model v{v_high} | LOW model v{v_low}")
    print(f"{'='*60}\n")

    return {
        "high_version": v_high,
        "low_version": v_low,
        "mae_high": round(mae_high, 6),
        "mae_low": round(mae_low, 6),
        "rmse_high": round(rmse_high, 6),
        "rmse_low": round(rmse_low, 6),
        "r2_high": round(r2_high, 4),
        "r2_low": round(r2_low, 4),
        "train_rows": len(X_train),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Range Model (HIGH/LOW prediction)")
    parser.add_argument("--warm", action="store_true", help="Warm-start from existing model")
    args = parser.parse_args()

    result = train(warm_start=args.warm)
    print(f"Final result: {result}")
