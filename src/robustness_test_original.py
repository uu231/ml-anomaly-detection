"""
robustness_test_original.py
使用原始模型（手工特征 + StandardScaler + LightGBM），取 train.csv 前 85% 时间步训练，
然后对原始验证集以及 11 个分布偏移变体进行预测，计算五个指标，
并将汇总结果保存到 outputs/robustness_test/robustness_metrics.txt。
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from lightgbm import LGBMClassifier, early_stopping as lgb_es

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_PATH, OUTPUT_DIR,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS, SEED,
    FEATURE_COLS,
)
from features import build_features


# ==================== 指标计算 ====================
def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc_pr = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "AUC-PR": auc_pr,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "FPR": fpr,
    }


def find_best_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_thresh = 0.5
    best_score = -1
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


# ==================== 变体生成 ====================
def get_feature_cols(df):
    return [col for col in df.columns if col.startswith("f") and col[1:].isdigit()]


def apply_scale_offset_noise(df, scale=1.0, offset=0.0, noise_std=0.0, seed=SEED):
    np.random.seed(seed)
    df_new = df.copy()
    features = get_feature_cols(df)
    for col in features:
        orig = df[col].values
        transformed = orig * scale + offset
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=len(orig))
            transformed += noise
        df_new[col] = transformed
    return df_new


VARIANTS = {
    "scale_08":       {"scale": 0.8, "offset": 0.0, "noise_std": 0.0},
    "scale_12":       {"scale": 1.2, "offset": 0.0, "noise_std": 0.0},
    "scale_15":       {"scale": 1.5, "offset": 0.0, "noise_std": 0.0},
    "offset_neg05":   {"scale": 1.0, "offset": -0.5, "noise_std": 0.0},
    "offset_pos05":   {"scale": 1.0, "offset": 0.5, "noise_std": 0.0},
    "offset_pos10":   {"scale": 1.0, "offset": 1.0, "noise_std": 0.0},
    "noise_02":       {"scale": 1.0, "offset": 0.0, "noise_std": 0.2},
    "noise_05":       {"scale": 1.0, "offset": 0.0, "noise_std": 0.5},
    "mix_easy":       {"scale": 1.1, "offset": 0.2, "noise_std": 0.05},
    "mix_medium":     {"scale": 1.3, "offset": -0.3, "noise_std": 0.15},
    "mix_hard":       {"scale": 1.5, "offset": 0.5, "noise_std": 0.3},
}


def generate_variants(df, output_dir, seed=SEED):
    os.makedirs(output_dir, exist_ok=True)
    variants_dfs = {}
    for name, params in VARIANTS.items():
        df_var = apply_scale_offset_noise(df, **params, seed=seed)
        path = os.path.join(output_dir, f"train_{name}.csv")
        df_var.to_csv(path, index=False)
        variants_dfs[name] = df_var
        print(f"  Variant '{name}' saved to {path}")
    return variants_dfs


# ==================== 训练与评估 ====================
def train_and_evaluate(train_df, variants_dir, output_dir):
    print("Building features on train data...")
    X_all, feat_names, predictors = build_features(
        train_df, show_progress=True, y_series=train_df["y"]
    )
    y = train_df["y"].values
    n = len(X_all)

    anomaly_start = int(np.where(y == 1)[0][0])
    split_idx = int(anomaly_start + (n - anomaly_start) * 0.85)
    X_train_raw = X_all[:split_idx]
    X_val_raw = X_all[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print("Training LightGBM...")
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb_es(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    y_prob_val = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = find_best_threshold(y_val, y_prob_val, metric="f1")
    print(f"Best threshold: {best_thresh:.4f} (F1={best_f1:.4f})")

    # --- 原始验证集指标 ---
    metrics_original = compute_all_metrics(y_val, y_prob_val, best_thresh)
    print("Original validation metrics:")
    for k, v in metrics_original.items():
        print(f"  {k}: {v:.4f}")

    # --- 变体生成与评估 ---
    print("\nGenerating shifted variants...")
    variants_dfs = generate_variants(train_df, variants_dir)

    all_results = {"original_val": metrics_original}
    for name, df_var in variants_dfs.items():
        X_var, _, _ = build_features(
            df_var, show_progress=False, trained_predictors=predictors
        )
        X_var_s = scaler.transform(X_var)
        y_true = df_var["y"].values
        y_prob = model.predict_proba(X_var_s)[:, 1]
        metrics = compute_all_metrics(y_true, y_prob, best_thresh)
        all_results[name] = metrics
        print(f"  {name}: F1={metrics['F1']:.4f}, FPR={metrics['FPR']:.4f}")

    # --- 保存汇总到 TXT ---
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "robustness_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Robustness Test Metrics (original model: StandardScaler + LightGBM)\n")
        f.write(f"Best threshold: {best_thresh:.4f}\n\n")
        header = f"{'Dataset':<20} {'AUC-PR':<10} {'F1':<10} {'Recall':<10} {'Precision':<10} {'FPR':<10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for ds_name, m in all_results.items():
            f.write(f"{ds_name:<20} {m['AUC-PR']:<10.4f} {m['F1']:<10.4f} "
                    f"{m['Recall']:<10.4f} {m['Precision']:<10.4f} {m['FPR']:<10.4f}\n")

    print(f"\nRobustness metrics saved to {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=TRAIN_PATH, help="Path to train.csv")
    parser.add_argument("--output_dir", default=os.path.join(OUTPUT_DIR, "robustness_test"),
                        help="Output directory for metrics")
    parser.add_argument("--variants_dir", default=os.path.join(OUTPUT_DIR, "robustness_test", "variants"),
                        help="Directory to save variant CSVs")
    args = parser.parse_args()

    train_df = pd.read_csv(args.input)
    print(f"Loaded train data: {train_df.shape}")

    train_and_evaluate(train_df, args.variants_dir, args.output_dir)


if __name__ == "__main__":
    main()
