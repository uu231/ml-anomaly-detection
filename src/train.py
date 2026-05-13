"""Model training with robust feature pipeline."""
import os
import numpy as np
import joblib
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from config import (
    LGBM_PARAMS, XGB_PARAMS, SCALE_POS_WEIGHT,
    N_TRIALS_OPTUNA, EARLY_STOPPING_ROUNDS, SEED,
    MODEL_DIR, TARGET_FPR, THRESHOLD_SEARCH_METRIC,
)
from validation import (
    purged_block_cv_splits, compute_metrics,
    find_best_threshold_robust,
)
from features import build_features  # 注意新的返回签名

warnings.filterwarnings("ignore")


# 原有 train_lgbm, train_xgboost, run_cv_validation 可保持不变（略）
# 这里只列出必须修改的部分

def train_final_model(train_df, tuned_params=None):
    """Train the final model on all training data with predictors."""
    y = train_df["y"].values
    # 构建特征并训练预测模型（需要 y_series）
    X, feat_names, predictors = build_features(train_df, show_progress=True, y_series=train_df["y"])

    # 时间分割：取前 85% 的正常数据作为阈值参考，剩余做验证
    n = len(X)
    anomaly_start = int(np.where(y == 1)[0][0])
    split_idx = int(anomaly_start + (n - anomaly_start) * 0.85)

    # 划分训练/验证
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[:split_idx])
    X_val = scaler.transform(X[split_idx:])
    y_tr, y_val = y[:split_idx], y[split_idx:]

    from lightgbm import LGBMClassifier, early_stopping as lgb_es
    params = tuned_params if tuned_params else LGBM_PARAMS.copy()
    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb_es(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    # 在验证集上搜索阈值
    y_prob_val = model.predict_proba(X_val)[:, 1]
    # 正常参考集：训练集中最后一段正常数据（可扩展）
    normal_mask_tr = (y_tr == 0)
    if normal_mask_tr.sum() > 0:
        # 取训练正常数据的预测分数（用验证集同理，但我们用训练集的不如用一段未用于训练的）
        # 这里使用验证集中的正常样本更合理，因为它们是“未来”数据
        y_val_normal = y_prob_val[y_val == 0]
        y_prob_normal_ref = y_val_normal
    else:
        y_prob_normal_ref = None

    best_thresh, best_score = find_best_threshold_robust(
        y_val, y_prob_val,
        y_prob_normal_only=y_prob_normal_ref,
        target_fpr=TARGET_FPR,
        metric=THRESHOLD_SEARCH_METRIC,
    )
    print(f"Best threshold: {best_thresh:.4f} (score={best_score:.4f})")

    return model, scaler, best_thresh, feat_names, predictors


def save_pipeline(model, scaler, threshold, feat_names, predictors, config, filepath):
    """Save complete pipeline including predictors."""
    pipeline = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "feature_names": feat_names,
        "predictors": predictors,   # 新增
        "config": config,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")


def load_pipeline(filepath):
    """Load a saved pipeline."""
    return joblib.load(filepath)