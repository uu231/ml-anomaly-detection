"""
train_robust_full.py
QuantileTransformer + Huber + 微调增强（noise_std=0.35, 全局偏移保持0.5）+ 误差极轻平滑
"""

import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score,
    precision_score, confusion_matrix,
)
from lightgbm import LGBMClassifier, early_stopping as lgb_es
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_PATH, OUTPUT_DIR, MODEL_DIR,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS, SEED,
    TARGET_FPR, THRESHOLD_SEARCH_METRIC,
)
from data_utils import load_train
from features import build_features          # Huber 回归版
from validation import find_best_threshold_robust


def augment_normal_samples(X, y, noise_std=0.35, scale_range=0.4, offset_range=1.0,
                           global_shift_prob=0.2, global_shift_mag=0.5, prob=0.5, seed=SEED):
    """仅对正常样本增强（无异常噪声），保持训练分布稳定"""
    rng = np.random.RandomState(seed)
    X_aug = X.copy()

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


def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc_pr = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    fpr = fp / (fp+tn) if (fp+tn)>0 else 0.0
    return {"AUC-PR": auc_pr, "F1": f1, "Recall": rec, "Precision": prec, "FPR": fpr}


def apply_scale_offset_noise(df, scale=1.0, offset=0.0, noise_std=0.0, seed=SEED):
    np.random.seed(seed)
    df_new = df.copy()
    features = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    for col in features:
        orig = df[col].values
        transformed = orig * scale + offset
        if noise_std > 0:
            transformed += np.random.normal(0, noise_std, size=len(orig))
        df_new[col] = transformed
    return df_new

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


def evaluate_variant(name, params, train_df, predictors, scaler, model, threshold, variants_dir):
    df_var = apply_scale_offset_noise(train_df, **params, seed=SEED)
    X_var, _, _ = build_features(df_var, show_progress=False, trained_predictors=predictors)
    X_var_s = scaler.transform(X_var)
    y_true = df_var["y"].values
    y_prob = model.predict_proba(X_var_s)[:, 1]
    metrics = compute_all_metrics(y_true, y_prob, threshold)
    return name, metrics


def main():
    print("Loading train data...")
    train_df = load_train()
    print("Building features (Huber prediction error)...")
    X_all, feat_names, predictors = build_features(train_df, show_progress=True, y_series=train_df["y"])
    y = train_df["y"].values
    n = len(X_all)

    anomaly_start = int(np.where(y == 1)[0][0])
    split_idx = int(anomaly_start + (n - anomaly_start) * 0.85)
    X_train_raw = X_all[:split_idx]
    X_val_raw = X_all[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]

    # QuantileTransformer（已验证稳定）
    scaler = QuantileTransformer(n_quantiles=min(1000, len(X_train_raw)), output_distribution='normal', random_state=SEED)
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # 微调增强：噪声强度从0.25提升到0.35，其他不变
    aug_params = {
        "noise_std": 0.35,            # 略微增大，匹配noise_02 (0.2) 和部分noise_05
        "scale_range": 0.4,
        "offset_range": 1.0,
        "global_shift_prob": 0.2,
        "global_shift_mag": 0.5,
        "prob": 0.5,
    }
    print(f"Augmenting normal samples with {aug_params}")
    X_train = augment_normal_samples(X_train, y_train, **aug_params, seed=SEED)

    # LightGBM 参数保持不变
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb_es(EARLY_STOPPING_ROUNDS, verbose=False)])

    # 鲁棒阈值
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_val_normal = y_prob_val[y_val == 0] if np.any(y_val == 0) else None
    best_thresh, best_score = find_best_threshold_robust(y_val, y_prob_val, y_prob_normal_only=y_val_normal, target_fpr=TARGET_FPR, metric=THRESHOLD_SEARCH_METRIC)
    print(f"Best threshold: {best_thresh:.4f} (score={best_score:.4f})")

    metrics_original = compute_all_metrics(y_val, y_prob_val, best_thresh)
    print("\nOriginal validation metrics:")
    for k,v in metrics_original.items(): print(f"  {k}: {v:.4f}")

    # 保存pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    pipeline = {"model": model, "scaler": scaler, "threshold": best_thresh, "feature_names": feat_names, "predictors": predictors, "config": {"lgbm_params": lgbm_params, "TARGET_FPR": TARGET_FPR, "aug_params": aug_params}}
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "robust_pipeline.pkl"))
    print("Pipeline saved.")

    # 变体评估
    print("\nEvaluating shifted variants...")
    variants_dir = "data/shifted_variants"
    os.makedirs(variants_dir, exist_ok=True)
    results = Parallel(n_jobs=4)(delayed(evaluate_variant)(name, params, train_df, predictors, scaler, model, best_thresh, variants_dir) for name, params in VARIANTS.items())
    all_results = {"original_val": metrics_original}
    for name, metrics in results:
        all_results[name] = metrics
        print(f"  {name}: F1={metrics['F1']:.4f}, FPR={metrics['FPR']:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_path = os.path.join(OUTPUT_DIR, "robustness_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Robustness Test Metrics (QuantileTransformer + noise_std=0.35)\n")
        f.write(f"Target FPR: {TARGET_FPR}\n")
        f.write(f"Best threshold: {best_thresh:.4f}\n\n")
        header = f"{'Dataset':<20} {'AUC-PR':<10} {'F1':<10} {'Recall':<10} {'Precision':<10} {'FPR':<10}"
        f.write(header + "\n" + "-"*len(header) + "\n")
        for ds_name, m in all_results.items():
            f.write(f"{ds_name:<20} {m['AUC-PR']:<10.4f} {m['F1']:<10.4f} {m['Recall']:<10.4f} {m['Precision']:<10.4f} {m['FPR']:<10.4f}\n")
    print(f"\nRobustness metrics saved to {txt_path}")

if __name__ == "__main__":
    main()