"""Purged Block CV and two-phase FPR validation, plus robust thresholding."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score,
    precision_score, confusion_matrix,
)
from data_utils import get_anomaly_clusters


def purged_block_cv_splits(y):
    """Create Purged Block CV splits based on anomaly clusters."""
    # ... 原代码完全不变 ...
    clusters = get_anomaly_clusters(y)
    n = len(y)
    anomaly_start = int(np.where(y.values == 1)[0][0])

    if len(clusters) < 2:
        splits = []
        for frac in [0.25, 0.5, 0.75]:
            split = anomaly_start + int((n - anomaly_start) * frac)
            splits.append((np.arange(0, split), np.arange(split, n)))
        return splits

    clusters = sorted(clusters, key=lambda c: c[0])
    splits = []

    for c_start, c_end in clusters:
        val_start = max(anomaly_start, c_start - 3)
        val_end = min(n - 1, c_end + 3)
        val_idx = np.arange(val_start, val_end + 1)

        train_idx = np.arange(0, val_end)
        train_idx = np.setdiff1d(train_idx, val_idx)

        if y.iloc[train_idx].sum() > 0 and y.iloc[val_idx].sum() > 0:
            splits.append((train_idx, val_idx))

    return splits


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "auc_pr": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def two_phase_fpr(model, X_normal, n_windows=4):
    """Evaluate false positive rate on normal-only data windows."""
    n = len(X_normal)
    window_size = n // n_windows
    fprs = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        X_w = X_normal[start:end]
        y_prob = model.predict_proba(X_w)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        fprs.append(y_pred.sum() / len(y_pred))

    return np.mean(fprs), np.std(fprs)


# ================= 新增：基于正常分布的阈值 =================
def threshold_from_normal_distribution(y_prob_normal, target_fpr=0.001):
    """
    根据正常样本的异常分数分布确定阈值。
    y_prob_normal: 仅包含正常样本的预测概率（或异常分数）
    target_fpr: 目标误报率
    """
    y_prob_normal = np.array(y_prob_normal)
    if len(y_prob_normal) == 0:
        return 0.5
    # 使用分位数：例如 1 - target_fpr 分位点作为阈值
    threshold = np.quantile(y_prob_normal, 1 - target_fpr)
    return float(threshold)


def find_best_threshold_robust(y_true, y_prob, y_prob_normal_only=None,
                               target_fpr=0.001, metric="f1"):
    """
    组合策略：先用正常分布找一个基准阈值（满足 target_fpr），
    然后在附近搜索提升验证集 F1/recall 的阈值（但不会让 FPR 超过 target_fpr*2）。
    """
    if y_prob_normal_only is not None and len(y_prob_normal_only) > 0:
        base_thresh = threshold_from_normal_distribution(y_prob_normal_only, target_fpr)
    else:
        base_thresh = 0.5

    # 搜索范围：base_thresh 的 ±0.2，但限制在 [0.01, 0.99]
    lower = max(0.01, base_thresh - 0.2)
    upper = min(0.99, base_thresh + 0.2)
    thresholds = np.linspace(lower, upper, 50)
    best_thresh = base_thresh
    best_score = -1

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        else:
            fpr = 0
        # 允许的 FPR 上限为目标值的 2 倍，且最小为 0.002
        max_fpr_allowed = max(target_fpr * 2, 0.002)
        if fpr > max_fpr_allowed:
            continue
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