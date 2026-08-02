"""
models/trainer.py — LightGBM model training with ML best practices:
  - Class-weight balancing (handles SELL/HOLD/BUY imbalance)
  - Early stopping with validation set (prevents overfitting)
  - Walk-forward time-series split (no future data leakage)
  - Weighted F1 score as primary evaluation metric
  - Warm-start retraining for online learning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter

from data.binance_fetcher import load_candles
from features.pipeline import build_features, get_feature_columns
from models.registry import save_model, load_active_model
from data.database import log_event
from config import LGBM_PARAMS, TIMEFRAMES


def _compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute inverse-frequency class weights so the model pays equal
    attention to rare signals (BUY/SELL) as to common ones (HOLD).
    """
    counter = Counter(y)
    total = len(y)
    n_classes = len(counter)
    weights = {cls: total / (n_classes * count) for cls, count in counter.items()}
    return weights


def _sample_weights(y: np.ndarray, class_weights: dict) -> np.ndarray:
    """Map class weights to per-sample weight array."""
    return np.array([class_weights[label] for label in y])


def train(timeframe: str, warm_start: bool = False) -> dict:
    """
    Train or retrain a LightGBM model for the given timeframe.

    Uses:
      - 70% train / 15% validation / 15% test (time-ordered, no shuffle)
      - Early stopping on validation set (stops when val loss stops improving)
      - Class-weight balancing to handle HOLD majority class
      - Saves model only if it beats previous version's F1 score
    """
    model_id = TIMEFRAMES[timeframe]["model_id"]
    print(f"\n[Trainer] {'Warm-start retraining' if warm_start else 'Initial training'} for {model_id}...")

    # ── Step 1: Load ALL available candles and build features ─────────────────
    raw_df = load_candles(timeframe, limit=None)  # load entire history from DB

    if raw_df.empty:
        raise ValueError(f"No candle data found for {timeframe}. Run binance_fetcher.py first.")

    print(f"[Trainer] Loaded {len(raw_df)} raw candles for {timeframe}.")
    feature_df = build_features(raw_df, timeframe, fit_scaler=True)

    feature_cols = get_feature_columns()
    available_features = [c for c in feature_cols if c in feature_df.columns]

    X = feature_df[available_features].values
    y = feature_df["label"].values.astype(int)

    print(f"[Trainer] Features: {len(available_features)} columns | Rows: {len(X)}")
    counts = Counter(y)
    total = len(y)
    print(f"[Trainer] Label distribution -> "
          f"SELL: {counts[0]} ({counts[0]/total*100:.1f}%) | "
          f"HOLD: {counts[1]} ({counts[1]/total*100:.1f}%) | "
          f"BUY:  {counts[2]} ({counts[2]/total*100:.1f}%)")

    # ── Step 2: Walk-forward split (time-ordered) ─────────────────────────────
    # 70% train | 15% validation (early stopping) | 15% test (final eval)
    n = len(X)
    train_end  = int(n * 0.70)
    val_end    = int(n * 0.85)

    X_train, y_train = X[:train_end],      y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"[Trainer] Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Step 3: Class-weight balancing ───────────────────────────────────────
    class_weights = _compute_class_weights(y_train)
    train_weights = _sample_weights(y_train, class_weights)
    print(f"[Trainer] Class weights -> SELL: {class_weights[0]:.3f} | "
          f"HOLD: {class_weights[1]:.3f} | BUY: {class_weights[2]:.3f}")

    # ── Step 4: Build LightGBM datasets ──────────────────────────────────────
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights,
                              feature_name=available_features)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # ── Step 5: Configure training params (adaptive to dataset size) ──────────
    n_train = len(X_train)
    # Scale complexity to training set size
    # Small datasets need simpler models to avoid early stopping at round 1
    if n_train < 5000:
        num_leaves       = 15
        min_child_samples = 10
        feature_fraction = 0.7
    elif n_train < 15000:
        num_leaves       = 31
        min_child_samples = 20
        feature_fraction = 0.8
    else:
        num_leaves       = 63
        min_child_samples = 50
        feature_fraction = 0.8

    params = {
        **LGBM_PARAMS,
        "metric":            "multi_logloss",
        "verbosity":         -1,
        "num_threads":       -1,       # use all CPU cores
        "num_leaves":        num_leaves,
        "min_child_samples": min_child_samples,
        "lambda_l1":         0.05,     # L1 regularization
        "lambda_l2":         0.05,     # L2 regularization
        "feature_fraction":  feature_fraction,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
    }
    print(f"[Trainer] Params -> num_leaves={num_leaves}, min_child_samples={min_child_samples}")


    # ── Step 6: Load existing model for warm-start if requested ──────────────
    init_model = None
    if warm_start:
        init_model = load_active_model(timeframe)
        if init_model:
            print(f"[Trainer] Loaded existing model for warm-start.")
        else:
            print(f"[Trainer] No existing model — doing fresh train instead.")

    # ── Step 7: Train with early stopping ────────────────────────────────────
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params       = params,
        train_set    = train_data,
        num_boost_round = 500,         # max trees — early stopping will cut this short
        valid_sets   = [val_data],
        callbacks    = callbacks,
        init_model   = init_model,
    )

    # ── Step 8: Evaluate on held-out test set ────────────────────────────────
    y_pred_proba = model.predict(X_test)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    accuracy    = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)

    print(f"\n[Trainer] === Test Results for {model_id} ===")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}  (primary metric)")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0)}")

    # Feature importance (top 10)
    importance = model.feature_importance(importance_type="gain")
    feat_importance = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)
    print(f"[Trainer] Top 10 features by gain:")
    for feat, imp in feat_importance[:10]:
        print(f"  {feat:<25} {imp:.1f}")

    # ── Step 9: Save model to registry ───────────────────────────────────────
    version = save_model(model, timeframe, accuracy=f1_weighted, train_rows=len(X_train))
    log_event("INFO",
              f"Trained model: f1_weighted={f1_weighted:.4f}, accuracy={accuracy:.4f}, rows={len(X_train)}",
              model_id=model_id)

    return {
        "model_id":     model_id,
        "version":      version,
        "accuracy":     round(accuracy, 4),
        "f1_weighted":  round(f1_weighted, 4),
        "f1_macro":     round(f1_macro, 4),
        "train_rows":   len(X_train),
        "best_iteration": model.best_iteration,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = train("1h", warm_start=False)
    print(f"\n[DONE] 1h: {result}\n{'='*60}")
