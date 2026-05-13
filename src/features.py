"""Temporal feature engineering pipeline with robust & prediction-error features."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from numpy.lib.stride_tricks import sliding_window_view

from config import (
    FEATURE_COLS, LAGS, DIFFS_1, DIFFS_2,
    ROLLING_WINDOWS, ROLLING_STATS, ROLLING_MINMAX_WINDOWS,
    ZSCORE_WINDOWS, EMA_SPANS,
    MAD_WINDOWS, PREDICTOR_LAGS, PREDICTOR_TRAIN_PROP, SEED,
)

# 基础特征总步数
TOTAL_BASE_STEPS = (
    1                                      # raw features
    + len(LAGS)                           # lag
    + len(DIFFS_1)                        # diff1
    + len(DIFFS_2)                        # diff2
    + len(ROLLING_WINDOWS) * len(ROLLING_STATS)   # rolling stats
    + len(ROLLING_MINMAX_WINDOWS) * 2             # rolling min/max
    + len(ZSCORE_WINDOWS)                          # z-score
    + len(MAD_WINDOWS)                              # MAD z-score
    + len(EMA_SPANS)                                 # EMA
)


def build_features(df, show_progress=True, y_series=None, trained_predictors=None):
    """Build temporal features + prediction error features from raw DataFrame.

    Args:
        df: input DataFrame with FEATURE_COLS
        show_progress: show tqdm progress
        y_series: (optional) labels for training predictors; if None, skip prediction features
        trained_predictors: (optional) pre-trained dict of predictors per feature
    Returns:
        X: feature matrix (numpy array)
        feature_names: list of column names
        predictors: dict of trained predictors (only if y_series is provided)
    """
    X = df[FEATURE_COLS].copy()
    X = X.ffill().fillna(0)

    feature_parts = [X.values]
    feature_names = list(X.columns)

    # ---------- 进度条总步数 ----------
    extra_steps = 0
    if y_series is not None or trained_predictors is not None:
        extra_steps = 1 + len(FEATURE_COLS) * 2   # 训练/推理 1步 + madz 33步 + errstd 33步
    total_steps = TOTAL_BASE_STEPS + extra_steps

    pbar = tqdm(total=total_steps, desc="  Building features", disable=not show_progress,
                bar_format="{desc}: {n}/{total} |{bar} | {elapsed}")

    # ================= 基础特征 =================
    # Lag features
    for lag in LAGS:
        lagged = X.shift(lag).ffill()
        lagged.columns = [f"{c}_lag{lag}" for c in X.columns]
        feature_parts.append(lagged.values)
        feature_names.extend(lagged.columns)
        pbar.update(1)

    # First-order diff
    for d in DIFFS_1:
        diffed = X.diff(d).fillna(0)
        diffed.columns = [f"{c}_diff1_{d}" for c in X.columns]
        feature_parts.append(diffed.values)
        feature_names.extend(diffed.columns)
        pbar.update(1)

    # Second-order diff
    for d in DIFFS_2:
        diff2 = X.diff(d).diff(d).fillna(0)
        diff2.columns = [f"{c}_diff2_{d}" for c in X.columns]
        feature_parts.append(diff2.values)
        feature_names.extend(diff2.columns)
        pbar.update(1)

    # Rolling statistics (mean/std)
    for w in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            rolled = getattr(X.rolling(window=w, min_periods=1), stat)()
            rolled = rolled.bfill()
            rolled.columns = [f"{c}_roll{stat}_{w}" for c in X.columns]
            feature_parts.append(rolled.values)
            feature_names.extend(rolled.columns)
            pbar.update(1)

    # Rolling min/max
    for w in ROLLING_MINMAX_WINDOWS:
        rolled_min = X.rolling(window=w, min_periods=1).min().bfill()
        rolled_min.columns = [f"{c}_rollmin_{w}" for c in X.columns]
        feature_parts.append(rolled_min.values)
        feature_names.extend(rolled_min.columns)
        pbar.update(1)

        rolled_max = X.rolling(window=w, min_periods=1).max().bfill()
        rolled_max.columns = [f"{c}_rollmax_{w}" for c in X.columns]
        feature_parts.append(rolled_max.values)
        feature_names.extend(rolled_max.columns)
        pbar.update(1)

    # Original Z-score
    for w in ZSCORE_WINDOWS:
        roll_mean = X.rolling(window=w, min_periods=1).mean().bfill()
        roll_std = X.rolling(window=w, min_periods=1).std().bfill()
        roll_std = roll_std.replace(0, 1e-8)
        zscore = ((X - roll_mean) / roll_std).fillna(0)
        zscore.columns = [f"{c}_zscore_{w}" for c in X.columns]
        feature_parts.append(zscore.values)
        feature_names.extend(zscore.columns)
        pbar.update(1)

    # MAD-based robust Z-score
    for w in MAD_WINDOWS:
        roll_median = X.rolling(window=w, min_periods=1).median().bfill()
        abs_diff = (X - roll_median).abs()
        roll_mad = abs_diff.rolling(window=w, min_periods=1).median().bfill()
        roll_mad = roll_mad.replace(0, 1e-8)
        mad_zscore = (0.6745 * (X - roll_median) / roll_mad).fillna(0)
        mad_zscore.columns = [f"{c}_madz_{w}" for c in X.columns]
        feature_parts.append(mad_zscore.values)
        feature_names.extend(mad_zscore.columns)
        pbar.update(1)

    # EMA
    for span in EMA_SPANS:
        ema = X.ewm(span=span, adjust=False, min_periods=1).mean().bfill()
        ema.columns = [f"{c}_ema_{span}" for c in X.columns]
        feature_parts.append(ema.values)
        feature_names.extend(ema.columns)
        pbar.update(1)

    # ================= 预测误差特征 =================
    predictors = None
    if trained_predictors is not None:
        # 推理模式
        pred_error_feats, pred_error_names = _compute_pred_errors_inference(
            X, trained_predictors, pbar
        )
        feature_parts.append(pred_error_feats)
        feature_names.extend(pred_error_names)
    elif y_series is not None:
        # 训练模式
        predictors, pred_error_feats, pred_error_names = _train_and_compute_pred_errors(
            X, y_series, pbar
        )
        feature_parts.append(pred_error_feats)
        feature_names.extend(pred_error_names)

    pbar.close()
    X_result = np.hstack(feature_parts)
    return X_result, feature_names, predictors


# ======================== 向量化工具 ========================
def _compute_errors_vectorized(series, lags, model):
    """用滑动窗口批量预测，返回与 series 等长的误差数组。"""
    max_lag = max(lags)
    n = len(series)
    if n <= max_lag:
        return np.zeros(n)

    # 窗口大小为 max_lag+1，这样每个窗口的最后一个是当前值
    windowed = sliding_window_view(series, window_shape=max_lag + 1)
    # windowed[t] = [series[t], series[t+1], ..., series[t+max_lag]]
    # 我们只能预测 t = max_lag ... n-1，对应的窗口索引为 0 ... n-max_lag-1
    windowed = windowed[:n - max_lag]   # 形状 (n - max_lag, max_lag + 1)

    # 滞后特征：取每个窗口的前 max_lag 列，然后逆序
    lagged = windowed[:, :max_lag]          # (n-max_lag, max_lag)
    lagged = lagged[:, ::-1]                # 逆序，现在第0列为series[t+max_lag-1]，第max_lag-1列为series[t]
    # 选取指定的滞后阶数 (lag -> 索引 lag-1)
    X_pred = lagged[:, [lag - 1 for lag in lags]]

    # 批量预测
    preds = model.predict(X_pred)          # 长度 n - max_lag
    errors = np.zeros(n)
    errors[max_lag:] = series[max_lag:] - preds
    return errors


def _make_ar_dataset(series, lags):
    """为训练预测模型创建滞后数据集。"""
    max_lag = max(lags)
    n = len(series)
    if n <= max_lag:
        return np.array([]).reshape(-1, 0), np.array([])

    windowed = sliding_window_view(series, window_shape=max_lag + 1)
    # 形状 (n - max_lag, max_lag + 1)
    X = windowed[:, :max_lag]                # 滞后特征
    y = windowed[:, max_lag]                 # 当前值
    # 逆序并选择指定滞后
    X = X[:, ::-1]                            # 逆序
    X = X[:, [lag - 1 for lag in lags]]      # 选列
    return X, y


# ======================== 训练预测器并生成误差 ========================
def _train_and_compute_pred_errors(X, y_series, pbar):
    """在正常数据上训练AR预测器，并计算整个序列的预测误差特征。"""
    normal_mask = (y_series.values == 0)
    if normal_mask.sum() < 50:
        raise ValueError("Not enough normal samples to train predictors.")

    normal_indices = np.where(normal_mask)[0]
    train_cut = int(len(normal_indices) * PREDICTOR_TRAIN_PROP)
    train_normal_idx = normal_indices[:train_cut]

    X_normal_train = X.iloc[train_normal_idx]
    predictors = {}
    all_errors = np.zeros((len(X), len(FEATURE_COLS)))

    # 对每个特征训练 AR 模型并计算误差
    for i, col in enumerate(FEATURE_COLS):
        series = X[col].values
        train_series = X_normal_train[col].values

        X_ar, y_ar = _make_ar_dataset(train_series, PREDICTOR_LAGS)
        if len(X_ar) < 10:
            mean_val = np.mean(train_series)
            all_errors[:, i] = series - mean_val
            predictors[col] = ("mean", mean_val)
            continue

        model = LinearRegression()
        model.fit(X_ar, y_ar)
        predictors[col] = model

        # 向量化计算整个序列的误差
        all_errors[:, i] = _compute_errors_vectorized(series, PREDICTOR_LAGS, model)

    pbar.update(1)

    # 构造误差特征 DataFrame
    error_df = pd.DataFrame(all_errors, columns=[f"{c}_pred_error" for c in FEATURE_COLS])
    error_feats = [error_df.values]
    error_names = list(error_df.columns)

    # 误差的滚动 MAD z‑score (窗口20)
    for col in error_df.columns:
        s = error_df[col]
        roll_median = s.rolling(window=20, min_periods=1).median().bfill()
        abs_diff = (s - roll_median).abs()
        roll_mad = abs_diff.rolling(window=20, min_periods=1).median().bfill()
        roll_mad = roll_mad.replace(0, 1e-8)
        mad_z = (0.6745 * (s - roll_median) / roll_mad).fillna(0)
        error_feats.append(mad_z.values.reshape(-1, 1))
        error_names.append(f"{col}_madz20")
        pbar.update(1)

    # 误差的滚动标准差 (窗口20)
    for col in error_df.columns:
        roll_std = error_df[col].rolling(window=20, min_periods=1).std().bfill()
        error_feats.append(roll_std.values.reshape(-1, 1))
        error_names.append(f"{col}_errstd20")
        pbar.update(1)

    return predictors, np.hstack(error_feats), error_names


# ======================== 推理模式 ========================
def _compute_pred_errors_inference(X, trained_predictors, pbar):
    """用已保存的 predictors 计算预测误差特征（推理时使用）。"""
    all_errors = np.zeros((len(X), len(FEATURE_COLS)))
    for i, col in enumerate(FEATURE_COLS):
        series = X[col].values
        pred_model = trained_predictors[col]
        if isinstance(pred_model, tuple) and pred_model[0] == "mean":
            mean_val = pred_model[1]
            all_errors[:, i] = series - mean_val
        else:
            all_errors[:, i] = _compute_errors_vectorized(series, PREDICTOR_LAGS, pred_model)

    pbar.update(1)

    error_df = pd.DataFrame(all_errors, columns=[f"{c}_pred_error" for c in FEATURE_COLS])
    error_feats = [error_df.values]
    error_names = list(error_df.columns)

    # 误差 MAD z‑score
    for col in error_df.columns:
        s = error_df[col]
        roll_median = s.rolling(window=20, min_periods=1).median().bfill()
        abs_diff = (s - roll_median).abs()
        roll_mad = abs_diff.rolling(window=20, min_periods=1).median().bfill()
        roll_mad = roll_mad.replace(0, 1e-8)
        mad_z = (0.6745 * (s - roll_median) / roll_mad).fillna(0)
        error_feats.append(mad_z.values.reshape(-1, 1))
        error_names.append(f"{col}_madz20")
        pbar.update(1)

    # 误差滚动标准差
    for col in error_df.columns:
        roll_std = error_df[col].rolling(window=20, min_periods=1).std().bfill()
        error_feats.append(roll_std.values.reshape(-1, 1))
        error_names.append(f"{col}_errstd20")
        pbar.update(1)

    return np.hstack(error_feats), error_names