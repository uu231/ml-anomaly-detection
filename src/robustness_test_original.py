"""
<<<<<<< HEAD
train_robust_full.py
使用 MAD + 预测误差(Huber) + 训练增强 + QuantileTransformer + 鲁棒阈值，
对 train.csv 的前 85% 训练，后 15% 验证，并评估 11 个偏移变体。
=======
robustness_test_original.py
使用原始模型（手工特征 + LightGBM），仅取 train.csv 前 85% 时间步训练，
然后对原始验证集以及多个分布偏移变体进行预测，计算五个指标，
并将汇总结果保存到 outputs/robustness_test/robustness_metrics.txt。
>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
"""

import os
import sys
<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
=======
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from lightgbm import LGBMClassifier, early_stopping as lgb_es

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

<<<<<<< HEAD
from config import (
    TRAIN_PATH, OUTPUT_DIR, MODEL_DIR,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS, SEED,
    TARGET_FPR, THRESHOLD_SEARCH_METRIC, FEATURE_COLS,
)
from data_utils import load_train
from features import build_features          # 已含 Huber 版预测误差
from validation import find_best_threshold_robust


# ===================== 强增强函数 =====================
def augment_normal_samples(X, y, noise_std=0.25, scale_range=0.4, offset_range=1.0,
                           global_shift_prob=0.2, global_shift_mag=0.5, prob=0.5, seed=SEED):
    """对正常样本施加逐样本扰动 + 概率性全局偏移"""
    rng = np.random.RandomState(seed)
    X_aug = X.copy()
    n_samples = len(X)

    mask_normal = y == 0
    idx_normal = np.where(mask_normal)[0]
    chosen = idx_normal[rng.rand(len(idx_normal)) < prob]
    for i in chosen:
        x = X[i].copy()
        scale = 1.0 + rng.uniform(-scale_range, scale_range)
        x = x * scale
        offset = rng.uniform(-offset_range, offset_range, size=x.shape)
        x = x + offset
        noise = rng.normal(0, noise_std, size=x.shape)
        x = x + noise
        X_aug[i] = x

    if rng.rand() < global_shift_prob:
        shift = rng.uniform(-global_shift_mag, global_shift_mag)
        X_aug += shift
        print(f"  Applied global shift: {shift:.4f}")

    return X_aug


# ===================== 指标计算 =====================
=======
from src.config import (
    TRAIN_PATH,
    LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    SEED,
)
from src.features import build_features


# ==================== 指标计算 ====================
>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc_pr = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
<<<<<<< HEAD
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"AUC-PR": auc_pr, "F1": f1, "Recall": rec, "Precision": prec, "FPR": fpr}


# ===================== 偏移变体 =====================
def get_feature_cols(df):
    return [col for col in df.columns if col.startswith("f") and col[1:].isdigit()]

=======
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "AUC-PR": auc_pr,
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
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


>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
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

<<<<<<< HEAD
=======

>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
VARIANTS = {
    "scale_08":  {"scale": 0.8, "offset": 0.0, "noise_std": 0.0},
    "scale_12":  {"scale": 1.2, "offset": 0.0, "noise_std": 0.0},
    "scale_15":  {"scale": 1.5, "offset": 0.0, "noise_std": 0.0},
    "offset_neg05": {"scale": 1.0, "offset": -0.5, "noise_std": 0.0},
    "offset_pos05": {"scale": 1.0, "offset": 0.5, "noise_std": 0.0},
    "offset_pos10": {"scale": 1.0, "offset": 1.0, "noise_std": 0.0},
    "noise_02":  {"scale": 1.0, "offset": 0.0, "noise_std": 0.2},
    "noise_05":  {"scale": 1.0, "offset": 0.0, "noise_std": 0.5},
    "mix_easy":  {"scale": 1.1, "offset": 0.2, "noise_std": 0.05},
    "mix_medium":{"scale": 1.3, "offset": -0.3, "noise_std": 0.15},
    "mix_hard":  {"scale": 1.5, "offset": 0.5, "noise_std": 0.3},
}


<<<<<<< HEAD
# ===================== 主流程 =====================
def main():
    # 1. 加载数据与特征
    print("Loading train data...")
    train_df = load_train()
    print("Building robust features (MAD + Huber prediction error)...")
    X_all, feat_names, predictors = build_features(
        train_df, show_progress=True, y_series=train_df["y"]
    )
    y = train_df["y"].values
    n = len(X_all)

    # 2. 时间划分
=======
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
    # --- 训练阶段 ---
    print("Building features on train data...")
    X_all, feat_names = build_features(train_df, show_progress=False)
    y = train_df["y"].values
    n = len(X_all)

>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
    anomaly_start = int(np.where(y == 1)[0][0])
    split_idx = int(anomaly_start + (n - anomaly_start) * 0.85)
    X_train_raw = X_all[:split_idx]
    X_val_raw = X_all[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]

<<<<<<< HEAD
    # 3. QuantileTransformer（映射到正态分布）
    scaler = QuantileTransformer(
        n_quantiles=min(1000, len(X_train_raw)),  # 分位数个数，不超过样本数
        output_distribution='normal',
        random_state=SEED
    )
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # 4. 强增强（在变换后的空间进行）
    aug_params = {
        "noise_std": 0.25,
        "scale_range": 0.4,
        "offset_range": 1.0,
        "global_shift_prob": 0.2,
        "global_shift_mag": 0.5,
        "prob": 0.5,
    }
    print(f"Augmenting normal training samples with {aug_params} ...")
    X_train = augment_normal_samples(X_train, y_train, **aug_params, seed=SEED)

    # 5. 训练 LightGBM（使用调整后的参数）
    lgbm_params = LGBM_PARAMS.copy()
    lgbm_params.update({
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    })
    print("Training LightGBM...")
    model = LGBMClassifier(**lgbm_params)
=======
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print("Training LightGBM...")
    model = LGBMClassifier(**LGBM_PARAMS)
>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb_es(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

<<<<<<< HEAD
    # 6. 鲁棒阈值
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_val_normal = y_prob_val[y_val == 0] if np.any(y_val == 0) else None
    best_thresh, best_score = find_best_threshold_robust(
        y_val, y_prob_val,
        y_prob_normal_only=y_val_normal,
        target_fpr=TARGET_FPR,
        metric=THRESHOLD_SEARCH_METRIC,
    )
    print(f"Best threshold: {best_thresh:.4f} (score={best_score:.4f})")

    # 7. 原始验证集指标
    metrics_original = compute_all_metrics(y_val, y_prob_val, best_thresh)
    print("\nOriginal validation metrics:")
    for k, v in metrics_original.items():
        print(f"  {k}: {v:.4f}")

    # 8. 保存 pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    pipeline = {
        "model": model,
        "scaler": scaler,
        "threshold": best_thresh,
        "feature_names": feat_names,
        "predictors": predictors,
        "config": {
            "lgbm_params": lgbm_params,
            "TARGET_FPR": TARGET_FPR,
            "aug_params": aug_params
        },
    }
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "robust_pipeline.pkl"))
    print("Pipeline saved.")

    # 9. 生成偏移变体并评估
    print("\nGenerating shifted variants...")
    variants_dir = "data/shifted_variants"
    os.makedirs(variants_dir, exist_ok=True)
    all_results = {"original_val": metrics_original}

    for name, params in VARIANTS.items():
        df_var = apply_scale_offset_noise(train_df, **params, seed=SEED)
        df_var.to_csv(os.path.join(variants_dir, f"train_{name}.csv"), index=False)

        # 变体特征构建 + 同样使用训练好的 scaler 变换
        X_var, _, _ = build_features(df_var, show_progress=False, trained_predictors=predictors)
=======
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
        X_var, _ = build_features(df_var, show_progress=False)
>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
        X_var_s = scaler.transform(X_var)
        y_true = df_var["y"].values
        y_prob = model.predict_proba(X_var_s)[:, 1]
        metrics = compute_all_metrics(y_true, y_prob, best_thresh)
        all_results[name] = metrics
        print(f"  {name}: F1={metrics['F1']:.4f}, FPR={metrics['FPR']:.4f}")

<<<<<<< HEAD
    # 10. 保存 TXT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_path = os.path.join(OUTPUT_DIR, "robustness_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Robustness Test Metrics (QuantileTransformer + strong aug + Huber)\n")
        f.write(f"Target FPR: {TARGET_FPR}\n")
=======
    # --- 保存汇总到 TXT ---
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "robustness_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Robustness Test Metrics (threshold fixed from original validation)\n")
>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
        f.write(f"Best threshold: {best_thresh:.4f}\n\n")
        header = f"{'Dataset':<20} {'AUC-PR':<10} {'F1':<10} {'Recall':<10} {'Precision':<10} {'FPR':<10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for ds_name, m in all_results.items():
            f.write(f"{ds_name:<20} {m['AUC-PR']:<10.4f} {m['F1']:<10.4f} {m['Recall']:<10.4f} {m['Precision']:<10.4f} {m['FPR']:<10.4f}\n")

    print(f"\nRobustness metrics saved to {txt_path}")


<<<<<<< HEAD
if __name__ == "__main__":
    main()
=======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=TRAIN_PATH, help="Path to train.csv")
    parser.add_argument("--output_dir", default="outputs/robustness_test", help="Output directory for metrics and variants")
    parser.add_argument("--variants_dir", default="data/shifted_variants", help="Directory to save variant CSVs")
    args = parser.parse_args()

    train_df = pd.read_csv(args.input)
    print(f"Loaded train data: {train_df.shape}")

    train_and_evaluate(train_df, args.variants_dir, args.output_dir)


if __name__ == "__main__":
    main()
>>>>>>> 9e10c4505e55177890f4b78e2ca992c47aea87f4
