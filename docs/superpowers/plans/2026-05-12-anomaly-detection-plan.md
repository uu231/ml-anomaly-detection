# Anomaly Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete pipeline for supervised anomaly detection on noisy time-series — feature engineering, Purged Block CV validation, LightGBM training with Optuna tuning, threshold optimization, and prediction generation for two test sets.

**Architecture:** Modular pipeline: config → data loading → feature engineering (lag/diff/rolling/EMA/z-score) → time-aware validation → LightGBM with Optuna → threshold tuning → prediction. All temporal constraints preserved; Task 2 uses same model as Task 1.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, LightGBM, XGBoost, Optuna, joblib

---

### Task 1: Project Config (`src/config.py`)

**Files:**
- Create: `src/config.py`
- Create: `src/__init__.py`

- [ ] **Step 1: Write config.py**

```python
"""All project constants, paths, and parameters."""
import os

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_SIMPLE_PATH = os.path.join(DATA_DIR, "test_simple.csv")
TEST_COMPLEX_PATH = os.path.join(DATA_DIR, "test_complex.csv")

# Random seed
SEED = 42

# Feature engineering
FEATURE_COLS = [f"f{i}" for i in range(1, 34)]
LAGS = [1, 2, 3, 5, 10]
DIFFS_1 = [1, 2, 3]
DIFFS_2 = [1, 2]
ROLLING_WINDOWS = [3, 5, 10, 20, 50]
ROLLING_STATS = ["mean", "std"]
ROLLING_MINMAX_WINDOWS = [5, 10, 20]
ZSCORE_WINDOWS = [10, 20, 50]
EMA_SPANS = [5, 10, 20]

# Validation
ANOMALY_START_IDX = 123390  # First row where y=1 appears
N_FOLDS = 4
FPR_WINDOWS = 4

# Model
SCALE_POS_WEIGHT = 240  # n_negative / n_positive
N_TRIALS_OPTUNA = 50
EARLY_STOPPING_ROUNDS = 50

# Default model params
LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}
```

- [ ] **Step 2: Create __init__.py**

```python
# src package
```

- [ ] **Step 3: Verify imports**

```bash
conda run -n ml_env python -c "import sys; sys.path.insert(0,'.'); from src.config import FEATURE_COLS, LAGS; print('config OK:', len(FEATURE_COLS), 'features')"
```

---

### Task 2: Data Utilities (`src/data_utils.py`)

**Files:**
- Create: `src/data_utils.py`

- [ ] **Step 1: Write data_utils.py**

```python
"""Load, validate, and preprocess data."""
import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_SIMPLE_PATH, TEST_COMPLEX_PATH, FEATURE_COLS, ANOMALY_START_IDX


def load_train():
    """Load and validate training data."""
    df = pd.read_csv(TRAIN_PATH)
    _validate_train(df)
    return df


def load_test_simple():
    """Load test_simple.csv."""
    return pd.read_csv(TEST_SIMPLE_PATH)


def load_test_complex():
    """Load test_complex.csv."""
    return pd.read_csv(TEST_COMPLEX_PATH)


def _validate_train(df):
    """Check train.csv integrity."""
    assert "y" in df.columns, "Missing label column 'y'"
    for c in FEATURE_COLS:
        assert c in df.columns, f"Missing feature column '{c}'"
    assert set(df["y"].unique()).issubset({0, 1}), "y must be 0/1"
    assert df["y"].sum() > 0, "No positive samples"


def validate_predictions(pred_df, test_df, name="predictions"):
    """Check prediction file format."""
    assert list(pred_df.columns) == ["y_pred"], f"{name}: column must be 'y_pred'"
    assert len(pred_df) == len(test_df), f"{name}: row count mismatch ({len(pred_df)} vs {len(test_df)})"
    assert set(pred_df["y_pred"].unique()).issubset({0, 1}), f"{name}: values must be 0/1"


def get_anomaly_clusters(y_series):
    """Identify contiguous anomaly clusters.
    
    Returns list of (start_idx, end_idx) for each cluster.
    """
    anomaly_idx = np.where(y_series.values == 1)[0]
    if len(anomaly_idx) == 0:
        return []
    
    clusters = []
    start = anomaly_idx[0]
    for i in range(1, len(anomaly_idx)):
        if anomaly_idx[i] - anomaly_idx[i-1] > 5:  # gap > 5 = new cluster
            clusters.append((start, anomaly_idx[i-1]))
            start = anomaly_idx[i]
    clusters.append((start, anomaly_idx[-1]))
    return clusters


def train_test_split_temporal(X, y, split_idx):
    """Split data at a given index, preserving temporal order."""
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    return X_train, X_val, y_train, y_val
```

- [ ] **Step 2: Test data loading**

```bash
conda run -n ml_env python -c "
import sys; sys.path.insert(0,'.')
from src.data_utils import load_train, load_test_simple, load_test_complex, get_anomaly_clusters
df = load_train()
print('Train:', df.shape, 'y=1:', df['y'].sum())
s = load_test_simple()
print('Test simple:', s.shape)
c = load_test_complex()
print('Test complex:', c.shape)
clusters = get_anomaly_clusters(df['y'])
print('Anomaly clusters:', len(clusters))
for i, (s_i, e_i) in enumerate(clusters):
    print(f'  Cluster {i}: [{s_i}, {e_i}], len={e_i-s_i+1}')
"
```

---

### Task 3: Feature Engineering (`src/features.py`)

**Files:**
- Create: `src/features.py`

- [ ] **Step 1: Write features.py**

```python
"""Temporal feature engineering pipeline."""
import numpy as np
import pandas as pd
from src.config import (
    FEATURE_COLS, LAGS, DIFFS_1, DIFFS_2,
    ROLLING_WINDOWS, ROLLING_STATS, ROLLING_MINMAX_WINDOWS,
    ZSCORE_WINDOWS, EMA_SPANS,
)


def build_features(df, fit_scaler=True, scaler=None):
    """Build temporal features from raw DataFrame.
    
    Args:
        df: DataFrame with columns f1..f33
        fit_scaler: if True, fit a new StandardScaler; if False, use provided scaler
        scaler: pre-fitted scaler to transform only
    
    Returns:
        X: feature matrix
        feature_names: list of column names
        scaler: fitted scaler (or None if not scaled here)
    """
    X = df[FEATURE_COLS].copy()
    X = X.ffill().fillna(0)
    
    feature_parts = [X.values]
    feature_names = list(X.columns)
    
    # Lag features
    for lag in LAGS:
        lagged = X.shift(lag).fillna(method="bfill")
        lagged.columns = [f"{c}_lag{lag}" for c in X.columns]
        feature_parts.append(lagged.values)
        feature_names.extend(lagged.columns)
    
    # First-order diff
    for d in DIFFS_1:
        diffed = X.diff(d).fillna(0)
        diffed.columns = [f"{c}_diff1_{d}" for c in X.columns]
        feature_parts.append(diffed.values)
        feature_names.extend(diffed.columns)
    
    # Second-order diff
    for d in DIFFS_2:
        diff2 = X.diff(1).diff(d - 1).fillna(0) if d > 1 else X.diff(2).fillna(0)
        diff2.columns = [f"{c}_diff2_{d}" for c in X.columns]
        feature_parts.append(diff2.values)
        feature_names.extend(diff2.columns)
    
    # Rolling window statistics
    for w in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            rolled = getattr(X.rolling(window=w, min_periods=1), stat)()
            rolled = rolled.fillna(method="bfill")
            rolled.columns = [f"{c}_roll{stat}_{w}" for c in X.columns]
            feature_parts.append(rolled.values)
            feature_names.extend(rolled.columns)
    
    for w in ROLLING_MINMAX_WINDOWS:
        rolled_min = X.rolling(window=w, min_periods=1).min().fillna(method="bfill")
        rolled_min.columns = [f"{c}_rollmin_{w}" for c in X.columns]
        feature_parts.append(rolled_min.values)
        feature_names.extend(rolled_min.columns)
        
        rolled_max = X.rolling(window=w, min_periods=1).max().fillna(method="bfill")
        rolled_max.columns = [f"{c}_rollmax_{w}" for c in X.columns]
        feature_parts.append(rolled_max.values)
        feature_names.extend(rolled_max.columns)
    
    # Rolling z-score
    for w in ZSCORE_WINDOWS:
        roll_mean = X.rolling(window=w, min_periods=1).mean().fillna(method="bfill")
        roll_std = X.rolling(window=w, min_periods=1).std().fillna(method="bfill").replace(0, 1e-8)
        zscore = ((X - roll_mean) / roll_std).fillna(0)
        zscore.columns = [f"{c}_zscore_{w}" for c in X.columns]
        feature_parts.append(zscore.values)
        feature_names.extend(zscore.columns)
    
    # EMA
    for span in EMA_SPANS:
        ema = X.ewm(span=span, adjust=False, min_periods=1).mean().fillna(method="bfill")
        ema.columns = [f"{c}_ema_{span}" for c in X.columns]
        feature_parts.append(ema.values)
        feature_names.extend(ema.columns)
    
    # Combine all features
    X_result = np.hstack(feature_parts)
    
    return X_result, feature_names


def get_feature_count():
    """Estimate total feature count."""
    n_base = len(FEATURE_COLS)  # raw
    n_lag = len(FEATURE_COLS) * len(LAGS)
    n_diff1 = len(FEATURE_COLS) * len(DIFFS_1)
    n_diff2 = len(FEATURE_COLS) * len(DIFFS_2)
    n_roll = len(FEATURE_COLS) * (len(ROLLING_WINDOWS) * len(ROLLING_STATS) + len(ROLLING_MINMAX_WINDOWS) * 2)
    n_zscore = len(FEATURE_COLS) * len(ZSCORE_WINDOWS)
    n_ema = len(FEATURE_COLS) * len(EMA_SPANS)
    return n_base + n_lag + n_diff1 + n_diff2 + n_roll + n_zscore + n_ema
```

- [ ] **Step 2: Test feature engineering**

```bash
conda run -n ml_env python -c "
import sys; sys.path.insert(0,'.')
from src.data_utils import load_train
from src.features import build_features, get_feature_count
df = load_train()
X, names = build_features(df)
print(f'Feature count: {len(names)} (estimated: {get_feature_count()})')
print(f'Samples: {X.shape[0]}')
print(f'First 5 names: {names[:5]}')
print(f'Last 5 names: {names[-5:]}')
print(f'NaN in X: {np.isnan(X).sum()}')
print('OK: no NaN' if np.isnan(X).sum() == 0 else 'FAIL: has NaN')
"
```

---

### Task 4: Validation (`src/validation.py`)

**Files:**
- Create: `src/validation.py`

- [ ] **Step 1: Write validation.py**

```python
"""Purged Block CV and two-phase FPR validation."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from src.data_utils import get_anomaly_clusters


def purged_block_cv_splits(y):
    """Create Purged Block CV splits based on anomaly clusters.
    
    Each fold leaves out one anomaly cluster as validation,
    training on all other data (preceding normal + all other clusters).
    
    Returns list of (train_indices, val_indices).
    """
    clusters = get_anomaly_clusters(y)
    n = len(y)
    
    if len(clusters) < 2:
        # Fallback: simple time split at anomaly region
        anomaly_start = int(np.where(y.values == 1)[0][0])
        splits = []
        for frac in [0.25, 0.5, 0.75]:
            split = anomaly_start + int((n - anomaly_start) * frac)
            splits.append((np.arange(0, split), np.arange(split, n)))
        return splits
    
    # Sort clusters by start index
    clusters = sorted(clusters, key=lambda c: c[0])
    
    splits = []
    for i, (c_start, c_end) in enumerate(clusters):
        # Validation: this cluster + buffer
        val_start = max(0, c_start - 3)
        val_end = min(n - 1, c_end + 3)
        val_idx = np.arange(val_start, val_end + 1)
        
        # Training: everything before val_end (minus val region)
        train_idx = np.arange(0, val_end)
        train_idx = np.setdiff1d(train_idx, val_idx)
        
        # Ensure train has some positives
        if y.iloc[train_idx].sum() > 0 and y.iloc[val_idx].sum() > 0:
            splits.append((train_idx, val_idx))
    
    return splits


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_pr": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "tn": int(confusion_matrix(y_true, y_pred).ravel()[0]) if confusion_matrix(y_true, y_pred).size == 4 else 0,
        "fp": int(confusion_matrix(y_true, y_pred).ravel()[1]) if confusion_matrix(y_true, y_pred).size == 4 else 0,
        "fn": int(confusion_matrix(y_true, y_pred).ravel()[2]) if confusion_matrix(y_true, y_pred).size == 4 else 0,
        "tp": int(confusion_matrix(y_true, y_pred).ravel()[3]) if confusion_matrix(y_true, y_pred).size == 4 else 0,
    }


def two_phase_fpr(model, X_normal, n_windows=4):
    """Evaluate false positive rate on normal-only data windows.
    
    X_normal: feature matrix from the normal (y=0 only) region.
    Returns mean FPR across windows.
    """
    n = len(X_normal)
    window_size = n // n_windows
    fprs = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        X_w = X_normal[start:end]
        y_prob = model.predict_proba(X_w)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        fpr = y_pred.sum() / len(y_pred)
        fprs.append(fpr)
    
    return np.mean(fprs), np.std(fprs)


def find_best_threshold(y_true, y_prob, metric="f1"):
    """Grid search for best decision threshold.
    
    Returns (best_threshold, best_score).
    """
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
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_thresh = t
    
    return best_thresh, best_score
```

- [ ] **Step 2: Test validation splits**

```bash
conda run -n ml_env python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from src.data_utils import load_train
from src.validation import purged_block_cv_splits
df = load_train()
splits = purged_block_cv_splits(df['y'])
print(f'Number of folds: {len(splits)}')
for i, (tr, val) in enumerate(splits):
    y_tr = df['y'].iloc[tr]
    y_val = df['y'].iloc[val]
    print(f'Fold {i}: train={len(tr)} (pos={y_tr.sum()}), val={len(val)} (pos={y_val.sum()})')
"
```

---

### Task 5: Model Training (`src/train.py`)

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write train.py**

```python
"""Model training with baseline comparison and Optuna tuning."""
import os
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import (
    LGBM_PARAMS, XGB_PARAMS, SCALE_POS_WEIGHT,
    N_TRIALS_OPTUNA, EARLY_STOPPING_ROUNDS, SEED,
    MODEL_DIR, ANOMALY_START_IDX,
)
from src.validation import purged_block_cv_splits, compute_metrics

warnings.filterwarnings("ignore")


def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    """Train a LightGBM classifier."""
    if params is None:
        params = LGBM_PARAMS.copy()
    
    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb_early_stopping(EARLY_STOPPING_ROUNDS)],
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train an XGBoost classifier."""
    if params is None:
        params = XGB_PARAMS.copy()
    
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest baseline."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic(X_train, y_train):
    """Train a Logistic Regression baseline."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


def lgb_early_stopping(rounds):
    """LightGBM early stopping callback."""
    try:
        import lightgbm as lgb
        return lgb.early_stopping(rounds, verbose=False)
    except ImportError:
        return None


def run_cv_validation(build_features_fn, train_df, model_factory, model_name, scaler=None):
    """Run Purged Block CV for a given model.
    
    Returns dict of fold metrics and aggregated results.
    """
    y = train_df["y"]
    splits = purged_block_cv_splits(y)
    
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Build features for this fold
        X_tr_raw = train_df.iloc[train_idx]
        X_val_raw = train_df.iloc[val_idx]
        
        # Build features with time-aware split (scaler fit on train only)
        X_tr, feat_names = build_features_fn(X_tr_raw)
        X_val, _ = build_features_fn(X_val_raw)
        
        # Scale
        if scaler is None:
            sc = StandardScaler()
        else:
            sc = scaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        
        y_tr = y.iloc[train_idx].values
        y_val = y.iloc[val_idx].values
        
        # Train
        model = model_factory(X_tr_s, y_tr, X_val_s, y_val)
        
        # Evaluate
        y_prob = model.predict_proba(X_val_s)[:, 1]
        metrics = compute_metrics(y_val, y_prob)
        metrics["fold"] = fold
        fold_results.append(metrics)
    
    # Aggregate
    agg = {}
    for key in ["auc_pr", "f1", "recall", "precision"]:
        vals = [r[key] for r in fold_results]
        agg[f"{key}_mean"] = np.mean(vals)
        agg[f"{key}_std"] = np.std(vals)
    
    agg["model"] = model_name
    agg["fold_results"] = fold_results
    
    return agg


def tune_with_optuna(X_train, y_train, X_val, y_val, n_trials=N_TRIALS_OPTUNA):
    """Bayesian hyperparameter optimization with Optuna."""
    try:
        import optuna
    except ImportError:
        print("Optuna not installed, using default params")
        return LGBM_PARAMS.copy()
    
    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 50, 500),
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)
    
    from sklearn.metrics import average_precision_score
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = LGBM_PARAMS.copy()
    best_params.update(study.best_params)
    return best_params


def train_final_model(build_features_fn, train_df, tuned_params=None):
    """Train the final model on all training data."""
    from sklearn.preprocessing import StandardScaler
    
    # Build features
    X_raw = train_df
    y = train_df["y"].values
    
    X, feat_names = build_features_fn(X_raw)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-aware split for early stopping (last 15% of anomaly region)
    n = len(X_scaled)
    split = int(ANOMALY_START_IDX + (n - ANOMALY_START_IDX) * 0.85)
    X_tr, X_val = X_scaled[:split], X_scaled[split:]
    y_tr, y_val = y[:split], y[split:]
    
    # Train
    params = tuned_params if tuned_params else LGBM_PARAMS.copy()
    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
    )
    
    return model, scaler, feat_names


def save_pipeline(model, scaler, threshold, feat_names, config, filepath):
    """Save complete pipeline for later use."""
    pipeline = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "feature_names": feat_names,
        "config": config,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")


def load_pipeline(filepath):
    """Load a saved pipeline."""
    return joblib.load(filepath)
```

- [ ] **Step 2: Verify training code syntax**

```bash
conda run -n ml_env python -c "import sys; sys.path.insert(0,'.'); exec(open('src/train.py').read()); print('train.py OK')"
```

---

### Task 6: Main Entry Point (`main.py`)

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

```python
#!/usr/bin/env python3
"""Main entry point for the anomaly detection pipeline.

Usage:
    python main.py --mode train
    python main.py --mode predict
    python main.py --mode all
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    TRAIN_PATH, TEST_SIMPLE_PATH, TEST_COMPLEX_PATH,
    MODEL_DIR, PRED_DIR, LGBM_PARAMS, SCALE_POS_WEIGHT, SEED,
)
from src.data_utils import load_train, load_test_simple, load_test_complex, validate_predictions
from src.features import build_features
from src.validation import find_best_threshold, compute_metrics, purged_block_cv_splits
from src.train import (
    train_lgbm, train_xgboost, train_random_forest, train_logistic,
    train_final_model, save_pipeline, load_pipeline,
    run_cv_validation,
)


def mode_train():
    """Full training pipeline."""
    np.random.seed(SEED)
    
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_train()
    print(f"  Train: {df.shape}, y=1: {df['y'].sum()} ({df['y'].mean():.4%})")
    
    # 2. Build features
    print("\n[2/5] Building features...")
    X, feat_names = build_features(df)
    y = df["y"].values
    print(f"  Features: {len(feat_names)} columns")
    
    # 3. Run baseline comparison
    print("\n[3/5] Running model comparison (Purged Block CV)...")
    
    from sklearn.preprocessing import StandardScaler
    
    def lgbm_factory(X_tr, y_tr, X_val, y_val):
        return train_lgbm(X_tr, y_tr, X_val, y_val, LGBM_PARAMS.copy())
    
    def xgb_factory(X_tr, y_tr, X_val, y_val):
        from src.config import XGB_PARAMS
        return train_xgboost(X_tr, y_tr, X_val, y_val, XGB_PARAMS.copy())
    
    def rf_factory(X_tr, y_tr, X_val, y_val):
        return train_random_forest(X_tr, y_tr)
    
    def lr_factory(X_tr, y_tr, X_val, y_val):
        return train_logistic(X_tr, y_tr)
    
    results = {}
    for name, factory in [
        ("LogisticRegression", lr_factory),
        ("RandomForest", rf_factory),
        ("XGBoost", xgb_factory),
        ("LightGBM", lgbm_factory),
    ]:
        try:
            r = run_cv_validation(build_features, df, factory, name)
            results[name] = r
            print(f"  {name:20s}: AUC-PR={r['auc_pr_mean']:.4f}+/-{r['auc_pr_std']:.4f}, "
                  f"F1={r['f1_mean']:.4f}, Recall={r['recall_mean']:.4f}")
        except Exception as e:
            print(f"  {name:20s}: FAILED - {e}")
    
    # 4. Train final model (LightGBM)
    print("\n[4/5] Training final LightGBM model on full data...")
    model, scaler, feat_names_out = train_final_model(build_features, df, LGBM_PARAMS.copy())
    
    # Determine threshold
    X_all, _ = build_features(df)
    X_all_s = scaler.transform(X_all)
    y_prob_all = model.predict_proba(X_all_s)[:, 1]
    
    # Use anomaly region for threshold tuning
    anomaly_start = 123390
    best_thresh, best_f1 = find_best_threshold(
        y[anomaly_start:], y_prob_all[anomaly_start:], metric="f1"
    )
    print(f"  Best threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    
    metrics = compute_metrics(y[anomaly_start:], y_prob_all[anomaly_start:], best_thresh)
    print(f"  Anomaly region metrics: AUC-PR={metrics['auc_pr']:.4f}, "
          f"F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}")
    
    # 5. Save pipeline
    print("\n[5/5] Saving pipeline...")
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    save_pipeline(model, scaler, best_thresh, feat_names_out, {"lgbm_params": LGBM_PARAMS}, model_path)
    
    print("\nTraining complete!")
    return results


def mode_predict():
    """Generate predictions for both test sets."""
    print("=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)
    
    # Load pipeline
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    pipeline = load_pipeline(model_path)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    threshold = pipeline["threshold"]
    print(f"  Loaded model, threshold={threshold:.3f}")
    
    # Predict test_simple
    print("\n[Task 1] Predicting test_simple.csv...")
    test_s = load_test_simple()
    X_s, _ = build_features(test_s)
    X_s_s = scaler.transform(X_s)
    y_prob_s = model.predict_proba(X_s_s)[:, 1]
    y_pred_s = (y_prob_s >= threshold).astype(int)
    
    pred_s = pd.DataFrame({"y_pred": y_pred_s})
    validate_predictions(pred_s, test_s, "test_simple")
    pred_path_s = os.path.join(PRED_DIR, "pred_simple.csv")
    os.makedirs(os.path.dirname(pred_path_s), exist_ok=True)
    pred_s.to_csv(pred_path_s, index=False)
    print(f"  Saved {pred_path_s} ({len(pred_s)} rows, {y_pred_s.sum()} positives)")
    
    # Predict test_complex
    print("\n[Task 2] Predicting test_complex.csv...")
    test_c = load_test_complex()
    X_c, _ = build_features(test_c)
    X_c_s = scaler.transform(X_c)
    y_prob_c = model.predict_proba(X_c_s)[:, 1]
    y_pred_c = (y_prob_c >= threshold).astype(int)
    
    pred_c = pd.DataFrame({"y_pred": y_pred_c})
    validate_predictions(pred_c, test_c, "test_complex")
    pred_path_c = os.path.join(PRED_DIR, "pred_complex.csv")
    pred_c.to_csv(pred_path_c, index=False)
    print(f"  Saved {pred_path_c} ({len(pred_c)} rows, {y_pred_c.sum()} positives)")
    
    print("\nPrediction complete!")


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all",
                        help="Pipeline mode (default: all)")
    args = parser.parse_args()
    
    if args.mode in ("train", "all"):
        mode_train()
    
    if args.mode in ("predict", "all"):
        mode_predict()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pipeline**

```bash
conda run -n ml_env python main.py --mode all
```
