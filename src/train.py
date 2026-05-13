"""Model training with baseline comparison and Optuna tuning."""
import os
import numpy as np
import joblib
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from src.config import (
    LGBM_PARAMS, XGB_PARAMS, SCALE_POS_WEIGHT,
    N_TRIALS_OPTUNA, EARLY_STOPPING_ROUNDS, SEED,
    MODEL_DIR,
)
from src.validation import purged_block_cv_splits, compute_metrics

warnings.filterwarnings("ignore")


def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    """Train a LightGBM classifier with early stopping."""
    from lightgbm import LGBMClassifier, early_stopping as lgb_es

    if params is None:
        params = LGBM_PARAMS.copy()

    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb_es(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train an XGBoost classifier."""
    from xgboost import XGBClassifier

    if params is None:
        params = XGB_PARAMS.copy()

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest baseline."""
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


def run_cv_validation(X, y, model_name, model_factory):
    """Run Purged Block CV for a given model.

    Uses pre-built feature matrix X (numpy array) and labels y.

    model_factory: callable(X_tr, y_tr, X_val, y_val) -> model
    """
    splits = purged_block_cv_splits(y)

    fold_results = []
    pbar = tqdm(splits, desc=f"  {model_name:20s}", bar_format="{desc} {n}/{total} |{bar}| {elapsed}")
    for fold, (train_idx, val_idx) in enumerate(pbar):
        X_tr = X[train_idx]
        X_val = X[val_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)

        y_tr = y.iloc[train_idx].values
        y_val = y.iloc[val_idx].values

        model = model_factory(X_tr_s, y_tr, X_val_s, y_val)

        y_prob = model.predict_proba(X_val_s)[:, 1]
        metrics = compute_metrics(y_val, y_prob)
        metrics["fold"] = fold
        fold_results.append(metrics)

        pbar.set_postfix({"AUC-PR": f"{metrics['auc_pr']:.3f}", "F1": f"{metrics['f1']:.3f}"})

    agg = {"model": model_name, "fold_results": fold_results}
    for key in ["auc_pr", "f1", "recall", "precision"]:
        vals = [r[key] for r in fold_results]
        agg[f"{key}_mean"] = np.mean(vals)
        agg[f"{key}_std"] = np.std(vals)

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

        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = LGBM_PARAMS.copy()
    best_params.update(study.best_params)
    return best_params


def _get_anomaly_start(y):
    """Detect the index of the first anomaly in the label series."""
    pos_idx = np.where(y.values == 1)[0]
    return int(pos_idx[0]) if len(pos_idx) > 0 else len(y)


def train_final_model(build_features_fn, train_df, tuned_params=None):
    """Train the final model on all training data."""
    y = train_df["y"].values
    X, feat_names = build_features_fn(train_df)

    # Fit scaler on train split only, then transform val — no leakage
    n = len(X)
    anomaly_start = _get_anomaly_start(train_df["y"])
    split = int(anomaly_start + (n - anomaly_start) * 0.85)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[:split])
    X_val = scaler.transform(X[split:])
    y_tr, y_val = y[:split], y[split:]

    from lightgbm import LGBMClassifier, early_stopping as lgb_es

    params = tuned_params if tuned_params else LGBM_PARAMS.copy()
    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb_es(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    return model, scaler, feat_names


def save_pipeline(model, scaler, threshold, feat_names, config, filepath):
    """Save complete pipeline."""
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
