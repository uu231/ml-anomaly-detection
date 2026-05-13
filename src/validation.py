"""Purged Block CV and two-phase FPR validation."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score,
    precision_score, confusion_matrix,
)
from src.data_utils import get_anomaly_clusters


def purged_block_cv_splits(y):
    """Create Purged Block CV splits based on anomaly clusters.

    Each fold leaves out one anomaly cluster as validation.
    """
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


def find_best_threshold(y_true, y_prob, metric="f1"):
    """Grid search for best decision threshold."""
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
