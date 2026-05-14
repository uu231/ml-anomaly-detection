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
ZSCORE_WINDOWS = [10, 20, 50]          # 保留但将在特征构建时用 MAD 版替代
EMA_SPANS = [5, 10, 20]


# 频域特征配置
STFT_WINDOW_SIZE = 32          # STFT 窗口长度（2的幂）
STFT_STEP = 16                 # 步长
STFT_TOP_FREQS = 5             # 保留能量最高的前5个频率分量
STFT_STATS = ["mean", "std"]   # 对频谱幅值的统计

# ----------------- 新增：稳健特征与预测误差 -----------------
MAD_WINDOWS = [10, 20, 50]              # MAD (中位数绝对偏差) 窗口
PREDICTOR_LAGS = [1, 2, 3, 5]          # 预测模型使用的滞后阶数（AR 模型）
PREDICTOR_TRAIN_PROP = 0.6             # 使用前60%的正常数据训练预测模型
NORMAL_TRAIN_FRAC = 0.85               # 阈值选择的正常参考集比例（按时间）
# ----------------------------------------------------------

# Validation
ANOMALY_START_IDX = None  # Auto-detected from data at runtime
N_FOLDS = 4
FPR_WINDOWS = 4
TARGET_FPR = 0.005                     # 正常数据上允许的最大误报率（目标）
THRESHOLD_SEARCH_METRIC = "f1"         # 最终仍以验证 F1 进行微调
# 预测误差相关（已在 features.py 中）
PREDICTOR_LAGS = [1, 2, 3, 5]
PREDICTOR_TRAIN_PROP = 0.6
# 训练增强的默认参数（可在训练脚本中覆盖）
AUG_NOISE_STD = 0.08
AUG_SCALE_RANGE = 0.03
AUG_OFFSET_RANGE = 0.05
AUG_PROB = 0.3

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