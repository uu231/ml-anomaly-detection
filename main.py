#!/usr/bin/env python3
"""Anomaly Detection Pipeline — Noisy Time-Series with Extreme Imbalance.

Usage:
    python main.py --mode train      # train and save model
    python main.py --mode predict    # generate prediction CSVs
    python main.py --mode all        # train + predict (default)
"""
import os
import sys
import time
import argparse

# Print immediately — imports (lightgbm, xgboost) are slow to load
print("Loading Anomaly Detection Pipeline...", flush=True)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Lightweight imports first
from src.config import (
    TRAIN_PATH, TEST_SIMPLE_PATH, TEST_COMPLEX_PATH,
    MODEL_DIR, PRED_DIR, LGBM_PARAMS, XGB_PARAMS, SEED, FEATURE_COLS,
)
from src.data_utils import (
    load_train, load_test_simple, load_test_complex, validate_predictions,
)
print("  - config & data utils loaded", flush=True)

from src.features import build_features, get_feature_count
print("  - feature engineering loaded", flush=True)

from src.validation import find_best_threshold_robust, compute_metrics
print("  - validation loaded", flush=True)

# Heavy imports (lightgbm, xgboost)
print("  - loading ML libraries (lightgbm, xgboost)...", flush=True)
from src.train import (
    train_lgbm, train_xgboost, train_random_forest, train_logistic,
    train_final_model, save_pipeline, load_pipeline, run_cv_validation,
)
print("  - training module loaded", flush=True)

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║     Robust Anomaly Detection in Noisy Time-Series Data      ║
║     LightGBM + Temporal Features + Purged Block CV          ║
╚══════════════════════════════════════════════════════════════╝"""


def _stamp(msg, start):
    """Print a step header with elapsed time."""
    elapsed = time.time() - start
    print(f"\n{'─' * 55}")
    print(f"  [{elapsed:5.0f}s] {msg}")
    print(f"{'─' * 55}")


def mode_train():
    """Full training pipeline."""
    np.random.seed(SEED)
    t0 = time.time()

    print(BANNER)
    print(f"  Mode: TRAIN")
    print(f"  Data: {TRAIN_PATH}")
    print(f"  Features: {len(FEATURE_COLS)} raw -> ~{get_feature_count()} engineered")
    print(f"  Model: LightGBM + XGBoost + RF + LR comparison")
    print(f"  Validation: Purged Block CV + Two-phase FPR")
    print(f"  Imbalance: 1:240, handled via scale_pos_weight + threshold tuning")

    # ── 1. Load ──────────────────────────────────────────────
    _stamp("STEP 1/5 — Loading training data", t0)
    df = load_train()
    n_pos = df["y"].sum()
    print(f"  Samples: {len(df):,}")
    print(f"  Features: {len(FEATURE_COLS)} (f1 ~ f33)")
    print(f"  Positive (y=1): {n_pos} ({n_pos/len(df)*100:.2f}%)")
    print(f"  Negative (y=0): {len(df)-n_pos} ({(len(df)-n_pos)/len(df)*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{int((len(df)-n_pos)/n_pos):,}")
    n_pos = df["y"].sum()
    first_anomaly = int(np.where(df["y"].values == 1)[0][0])
    print(f"  First anomaly at index {first_anomaly} (last {(len(df)-first_anomaly)/len(df)*100:.0f}% of data)")

    # ── 2. Features ──────────────────────────────────────────
    _stamp("STEP 2/5 — Building temporal features", t0)
    print(f"  Lag:           {[1,2,3,5,10]}")
    print(f"  Diff1:         {[1,2,3]}")
    print(f"  Diff2:         {[1,2]}")
    print(f"  Rolling stats: mean/std (w=3,5,10,20,50) + min/max (w=5,10,20)")
    print(f"  Z-score:       w=10,20,50")
    print(f"  EMA:           span=5,10,20")
    print(f"  Target: ~{get_feature_count()} features")
    X, feat_names, _ = build_features(df)
    y = df["y"].values
    print(f"  Built: {X.shape[1]} features x {X.shape[0]:,} samples")
    print(f"  Memory: {X.nbytes / 1024**2:.0f} MB")

    # ── 3. Model comparison ──────────────────────────────────
    _stamp("STEP 3/5 — Model comparison (Purged Block CV)", t0)
    print(f"  Strategy: Leave-one-anomaly-cluster-out")
    print(f"  Anomaly clusters: ~19 clusters of ~30 samples each")
    print(f"  Each fold: train on all but one cluster, validate on held-out cluster")

    results = {}
    models_to_run = [
        ("LightGBM",
         lambda Xt, yt, Xv, yv: train_lgbm(Xt, yt, Xv, yv, LGBM_PARAMS.copy())),
        # GPU training later:
        # ("XGBoost",
        #  lambda Xt, yt, Xv, yv: train_xgboost(Xt, yt, Xv, yv, XGB_PARAMS.copy())),
        # ("RandomForest",
        #  lambda Xt, yt, Xv, yv: train_random_forest(Xt, yt)),
        # ("LogisticRegression",
        #  lambda Xt, yt, Xv, yv: train_logistic(Xt, yt)),
    ]

    for name, factory in models_to_run:
        try:
            r = run_cv_validation(X, df["y"], name, factory)
            results[name] = r
            print(f"  → {name:20s} | AUC-PR={r['auc_pr_mean']:.4f} ±{r['auc_pr_std']:.4f} | "
                  f"F1={r['f1_mean']:.4f} | Recall={r['recall_mean']:.4f} | "
                  f"Precision={r['precision_mean']:.4f}")
        except Exception as e:
            print(f"  → {name:20s} | FAILED: {e}")

    # Print comparison table
    print(f"\n  {'Model':20s} {'AUC-PR':>8s} {'F1':>8s} {'Recall':>8s} {'Precision':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["auc_pr_mean"]):
        print(f"  {name:20s} {r['auc_pr_mean']:8.4f} {r['f1_mean']:8.4f} "
              f"{r['recall_mean']:8.4f} {r['precision_mean']:8.4f}")

    # ── 4. Final model ───────────────────────────────────────
    _stamp("STEP 4/5 — Training final model on ALL data", t0)
    print(f"  Using best model: LightGBM")
    print(f"  Training on full train.csv ({len(df):,} rows)")
    model, scaler, best_thresh, feat_names_out, predictors = train_final_model(df)

    # Rebuild features on full data with trained predictors for metrics display
    print(f"\n  Computing full-data metrics...")
    X_all, _, _ = build_features(df, trained_predictors=predictors)
    X_all_s = scaler.transform(X_all)
    y_prob_all = model.predict_proba(X_all_s)[:, 1]

    anomaly_start = int(np.where(df["y"].values == 1)[0][0])
    print(f"  Anomaly region: [{anomaly_start}, {len(y)}) = {len(y)-anomaly_start} samples")

    metrics = compute_metrics(y[anomaly_start:], y_prob_all[anomaly_start:], best_thresh)
    print(f"  Best threshold: {best_thresh:.4f} (from train_final_model)")
    print(f"  ┌──────────┬────────┐")
    print(f"  │ Metric   │ Value  │")
    print(f"  ├──────────┼────────┤")
    print(f"  │ AUC-PR   │ {metrics['auc_pr']:.4f} │")
    print(f"  │ F1       │ {metrics['f1']:.4f} │")
    print(f"  │ Recall   │ {metrics['recall']:.4f} │")
    print(f"  │ Precision│ {metrics['precision']:.4f} │")
    print(f"  │ TP / FP  │ {metrics['tp']:4d} / {metrics['fp']:4d} │")
    print(f"  │ FN / TN  │ {metrics['fn']:4d} / {metrics['tn']:4d} │")
    print(f"  └──────────┴────────┘")

    # Two-phase FPR: evaluate on normal-only regions
    from src.validation import two_phase_fpr
    normal_region = X_all_s[:anomaly_start]
    fpr_mean, fpr_std = two_phase_fpr(model, normal_region, n_windows=4)
    print(f"\n  Two-phase FPR check (first 90% normal-only data):")
    print(f"    FPR = {fpr_mean:.4f} +/- {fpr_std:.4f} "
          f"({fpr_mean*100:.2f}% of normal samples incorrectly flagged)")

    # ── 5. Save ──────────────────────────────────────────────
    _stamp("STEP 5/5 — Saving pipeline", t0)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    save_pipeline(model, scaler, best_thresh, feat_names_out, predictors,
                  {"lgbm_params": LGBM_PARAMS}, model_path)

    _stamp("TRAINING COMPLETE", t0)
    print(f"  Total time: {time.time() - t0:.0f}s")
    print(f"  Model saved to: {model_path}")
    return results


def mode_predict():
    """Generate predictions for both test sets using saved model."""
    t0 = time.time()
    print(BANNER)
    print(f"  Mode: PREDICT")

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    if not os.path.exists(model_path):
        print(f"\n  ERROR: No saved model at {model_path}")
        print(f"  Run 'python main.py --mode train' first.")
        return

    # Load
    _stamp("Loading trained pipeline", t0)
    pipeline = load_pipeline(model_path)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    threshold = pipeline["threshold"]
    predictors = pipeline.get("predictors", None)
    print(f"  Model: LightGBM ({model.best_iteration_} trees)")
    print(f"  Threshold: {threshold:.3f} (fixed — same for both tasks)")
    if predictors is not None:
        print(f"  Predictors: {len(predictors)} AR models loaded")

    # Task 1
    _stamp("Task 1 — Predicting test_simple.csv (similar distribution)", t0)
    test_s = load_test_simple()
    print(f"  Loading: {len(test_s):,} rows x {len(FEATURE_COLS)} features")
    X_s, _, _ = build_features(test_s, trained_predictors=predictors)
    X_s_s = scaler.transform(X_s)
    y_pred_s = (model.predict_proba(X_s_s)[:, 1] >= threshold).astype(int)
    n_pos_s = y_pred_s.sum()

    pred_s = pd.DataFrame({"y_pred": y_pred_s})
    validate_predictions(pred_s, test_s, "test_simple")
    pred_path_s = os.path.join(PRED_DIR, "pred_simple.csv")
    os.makedirs(os.path.dirname(pred_path_s), exist_ok=True)
    pred_s.to_csv(pred_path_s, index=False)
    print(f"  Predicted positives: {n_pos_s} / {len(pred_s):,} ({n_pos_s/len(pred_s)*100:.2f}%)")
    print(f"  Saved: {pred_path_s}")

    # Task 2
    _stamp("Task 2 — Predicting test_complex.csv (complex scenario)", t0)
    print(f"  NOTE: Using SAME model, SAME threshold as Task 1 — no retraining!")
    test_c = load_test_complex()
    print(f"  Loading: {len(test_c):,} rows x {len(FEATURE_COLS)} features")
    X_c, _, _ = build_features(test_c, trained_predictors=predictors)
    X_c_s = scaler.transform(X_c)
    y_pred_c = (model.predict_proba(X_c_s)[:, 1] >= threshold).astype(int)
    n_pos_c = y_pred_c.sum()

    pred_c = pd.DataFrame({"y_pred": y_pred_c})
    validate_predictions(pred_c, test_c, "test_complex")
    pred_path_c = os.path.join(PRED_DIR, "pred_complex.csv")
    pred_c.to_csv(pred_path_c, index=False)
    print(f"  Predicted positives: {n_pos_c} / {len(pred_c):,} ({n_pos_c/len(pred_c)*100:.2f}%)")
    print(f"  Saved: {pred_path_c}")

    _stamp("PREDICTION COMPLETE", t0)
    print(f"  Total time: {time.time() - t0:.0f}s")
    print(f"\n  Summary:")
    print(f"    test_simple:  {n_pos_s:6d} positives / {len(test_s):,} rows")
    print(f"    test_complex: {n_pos_c:6d} positives / {len(test_c):,} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Robust Anomaly Detection in Noisy Time-Series Data")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all",
                        help="Pipeline mode (default: all)")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        mode_train()

    if args.mode in ("predict", "all"):
        mode_predict()


if __name__ == "__main__":
    main()
