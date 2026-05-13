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
ANOMALY_START_IDX = None  # Auto-detected from data at runtime
N_FOLDS = 4
FPR_WINDOWS = 4

# Model
SCALE_POS_WEIGHT = 240
N_TRIALS_OPTUNA = 50
EARLY_STOPPING_ROUNDS = 50

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
    "scale_pos_weight": SCALE_POS_WEIGHT,
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
