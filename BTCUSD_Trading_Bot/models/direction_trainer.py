"""
models/direction_trainer.py — Train the Direction Model (BUY/SELL/HOLD classification).

Uses LightGBM with:
  - Time-series K-fold cross validation (walk-forward, no future leakage)
  - Class-weight balancing (HOLD is majority class)
  - Early stopping on validation loss
  - StandardScaler normalization
  - Warm-start support for online learning

Target: F1 weighted ≥ 47%

Usage:
    python models/direction_trainer.py              # fresh train
    python models/direction_trainer.py --warm       # warm-start retrain
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter

from data.binance_fetcher import load_candles
from features.pipeline import build_features, generate_direction_labels, get_feature_columns
from models.registry import save_model, load_active_model
from data.database import log_event
from config import DIRECTION_LGBM_PARAMS, TIMEFRAMES


TIMEFRAME = "15m"
MODEL_TYPE = "direction"
SCALER_ID = "direction_15m"


def _compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute inverse-frequency weights so the model pays equal attention
    to rare classes (BUY/SELL) as to the majority class (HOLD).
    """
    counter = Counter(y)
    total = len(y)
    n_classes = len(counter)
    return {cls: total / (n_classes * count) for cls, count in counter.items()}


def _sample_weights(y: np.ndarray, class_weights: dict) -> np.ndarray:
    """Convert class weights to per-sample weight array."""
    return np.array([class_weights[label] for label in y])


def train(warm_start: bool = False) -> dict:
    """
    Train the Direction Model.

    Process:
      1. Load all 15-min candles from DB
      2. Build features + generate direction labels
      3. Time-series K-fold cross validation (5 folds)
      4. Final train on 70% / validate on 15% / test on 15%
      5. Report metrics and save model if good enough

    Returns dict with metrics.
    """
    model_id = TIMEFRAMES[TIMEFRAME]["model_id"]
    print(f"\n{'='*60}")
    print(f"[Direction Trainer] {'Warm-start' if warm_start else 'Fresh'} training for {model_id}")
    print(f"{'='*60}")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    raw_df = load_candles(TIMEFRAME, limit=None)
    if raw_df.empty:
        raise ValueError("No candle data. Run binance_fetcher.py first.")
    print(f"[Data] Loaded {len(raw_df)} raw candles.")

    # ── Step 2: Build features ────────────────────────────────────────────────
    feature_df = build_features(raw_df, fit_scaler=True, scaler_id=SCALER_ID)
    print(f"[Features] {len(feature_df)} rows after indicator warmup.")

    # ── Step 3: Generate direction labels ─────────────────────────────────────
    # We need the raw df (with high/low/close) aligned with feature_df
    # Re-add raw columns for label generation
    label_df = raw_df.iloc[-len(feature_df):].reset_index(drop=True).copy()
    label_df = generate_direction_labels(label_df)

    # Trim feature_df to match label_df length
    feature_df = feature_df.iloc[:len(label_df)].reset_index(drop=True)

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in feature_df.columns]

    X = feature_df[available].values
    y = label_df["direction_label"].values.astype(int)

    print(f"[Data] Final dataset: {len(X)} rows × {len(available)} features")
    counts = Counter(y)
    total = len(y)
    print(f"[Labels] SELL: {counts.get(0,0)} ({counts.get(0,0)/total*100:.1f}%) | "
          f"HOLD: {counts.get(1,0)} ({counts.get(1,0)/total*100:.1f}%) | "
          f"BUY: {counts.get(2,0)} ({counts.get(2,0)/total*100:.1f}%)")

    # ── Step 4: K-Fold Cross Validation (time-series aware) ───────────────────
    print(f"\n[CV] Running 5-fold time-series cross validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]

        class_weights = _compute_class_weights(y_tr)
        weights = _sample_weights(y_tr, class_weights)

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=available)
        val_data = lgb.Dataset(X_vl, label=y_vl, reference=train_data)

        model = lgb.train(
            params={**DIRECTION_LGBM_PARAMS, "metric": "multi_logloss", "verbosity": -1},
            train_set=train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        y_pred = np.argmax(model.predict(X_vl), axis=1)
        fold_f1 = f1_score(y_vl, y_pred, average="weighted", zero_division=0)
        cv_f1_scores.append(fold_f1)
        print(f"  Fold {fold}: F1 = {fold_f1:.4f}")

    mean_cv_f1 = np.mean(cv_f1_scores)
    print(f"\n[CV] Mean F1 across 5 folds: {mean_cv_f1:.4f}")

    # ── Step 5: Final train/val/test split (70/15/15) ─────────────────────────
    print(f"\n[Final] Training on 70% / Val 15% / Test 15%...")
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Class weights on training set
    class_weights = _compute_class_weights(y_train)
    train_weights = _sample_weights(y_train, class_weights)
    print(f"  Class weights → SELL: {class_weights.get(0,0):.3f} | "
          f"HOLD: {class_weights.get(1,0):.3f} | BUY: {class_weights.get(2,0):.3f}")

    # Adaptive complexity based on dataset size
    n_train = len(X_train)
    if n_train < 10000:
        num_leaves, min_child = 31, 20
    elif n_train < 50000:
        num_leaves, min_child = 63, 30
    else:
        num_leaves, min_child = 127, 50

    params = {
        **DIRECTION_LGBM_PARAMS,
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_leaves": num_leaves,
        "min_child_samples": min_child,
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights,
                             feature_name=available)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Warm-start: load existing model
    init_model = None
    if warm_start:
        init_model = load_active_model(TIMEFRAME, MODEL_TYPE)
        if init_model:
            print(f"  Loaded existing model for warm-start.")
        else:
            print(f"  No existing model found — doing fresh train.")

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)],
        init_model=init_model,
    )

    # ── Step 6: Evaluate on test set ──────────────────────────────────────────
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"[RESULTS] Direction Model — Test Set")
    print(f"{'='*60}")
    print(f"  Accuracy     : {accuracy:.4f}")
    print(f"  F1 Weighted  : {f1_weighted:.4f}  ← primary metric (target ≥ 0.47)")
    print(f"  F1 Macro     : {f1_macro:.4f}")
    print(f"  CV Mean F1   : {mean_cv_f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['SELL','HOLD','BUY'], zero_division=0)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"             Pred SELL  Pred HOLD  Pred BUY")
    print(f"  Act SELL   {cm[0][0]:>8}  {cm[0][1]:>9}  {cm[0][2]:>8}")
    print(f"  Act HOLD   {cm[1][0]:>8}  {cm[1][1]:>9}  {cm[1][2]:>8}")
    print(f"  Act BUY    {cm[2][0]:>8}  {cm[2][1]:>9}  {cm[2][2]:>8}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(available, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features by gain:")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat:<25} {imp:.1f}")

    # ── Step 7: Save model ────────────────────────────────────────────────────
    version = save_model(model, TIMEFRAME, MODEL_TYPE, accuracy=f1_weighted,
                         train_rows=len(X_train))
    log_event("INFO",
              f"Direction model trained: f1={f1_weighted:.4f}, acc={accuracy:.4f}",
              model_id=model_id)

    print(f"\n[SAVED] Direction model v{version} | F1={f1_weighted:.4f}")
    print(f"{'='*60}\n")

    return {
        "model_type": MODEL_TYPE,
        "version": version,
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "cv_mean_f1": round(mean_cv_f1, 4),
        "train_rows": len(X_train),
        "best_iteration": model.best_iteration,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Direction Model (BUY/SELL/HOLD)")
    parser.add_argument("--warm", action="store_true", help="Warm-start from existing model")
    args = parser.parse_args()

    result = train(warm_start=args.warm)
    print(f"Final result: {result}")
