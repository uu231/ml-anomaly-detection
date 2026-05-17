"""Temporal feature engineering pipeline with robust & prediction-error features (Huber) + frequency stats."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import HuberRegressor
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import periodogram

from config import (
    FEATURE_COLS, LAGS, DIFFS_1, DIFFS_2,
    ROLLING_WINDOWS, ROLLING_STATS, ROLLING_MINMAX_WINDOWS,
    ZSCORE_WINDOWS, EMA_SPANS,
    MAD_WINDOWS, PREDICTOR_LAGS, PREDICTOR_TRAIN_PROP, SEED,
)

# 基础特征总步数（不含频域和预测误差）
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

# 频域特征步数（每个特征3个统计量）
FREQ_STEPS = len(FEATURE_COLS)


def get_feature_count(with_predictors=True):
    """Return approximate number of engineered features."""
    n_time = (1 + len(LAGS) + len(DIFFS_1) + len(DIFFS_2)
              + len(ROLLING_WINDOWS) * len(ROLLING_STATS)
              + len(ROLLING_MINMAX_WINDOWS) * 2
              + len(ZSCORE_WINDOWS) + len(MAD_WINDOWS)
              + len(EMA_SPANS)) * len(FEATURE_COLS)
    n_freq = len(FEATURE_COLS) * 3
    n_pred = len(FEATURE_COLS) * 3 if with_predictors else 0
    return n_time + n_freq + n_pred

def build_features(df, show_progress=True, y_series=None, trained_predictors=None):
    """Build temporal + frequency + prediction error features (Huber)."""
    X = df[FEATURE_COLS].copy()
    X = X.ffill().fillna(0)

    feature_parts = [X.values]
    feature_names = list(X.columns)

    # 计算总步数
    extra_steps = 0
    if y_series is not None or trained_predictors is not None:
        extra_steps = 1 + len(FEATURE_COLS) * 2   # 预测误差 + madz + errstd
    total_steps = TOTAL_BASE_STEPS + FREQ_STEPS + extra_steps

    pbar = tqdm(total=total_steps, desc="  Building features", disable=not show_progress,
                bar_format="{desc}: {n}/{total} |{bar} | {elapsed}")

    # ================= 1. 基础时域特征 =================
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

    # ================= 2. 频域特征（在 QuantileTransformer 之前计算） =================
    _add_frequency_features(X, feature_parts, feature_names, pbar)

    # ================= 3. 预测误差特征（Huber回归） =================
    predictors = None
    if trained_predictors is not None:
        pred_error_feats, pred_error_names = _compute_pred_errors_inference(
            X, trained_predictors, pbar
        )
        feature_parts.append(pred_error_feats)
        feature_names.extend(pred_error_names)
    elif y_series is not None:
        predictors, pred_error_feats, pred_error_names = _train_and_compute_pred_errors(
            X, y_series, pbar
        )
        feature_parts.append(pred_error_feats)
        feature_names.extend(pred_error_names)

    pbar.close()
    X_result = np.hstack(feature_parts)
    return X_result, feature_names, predictors


def _add_frequency_features(X, feat_parts, feat_names, pbar):
    """
    对每个特征，每隔 step 点计算一次周期图，提取均值、标准差、谱熵，
    并前向填充到整个窗口，完全避免未来信息泄露。
    """
    step = 50          # 滑动步长
    window = 64        # 周期图窗口
    n_samples = X.shape[0]
    stats_labels = ["freq_mean", "freq_std", "freq_entropy"]

    for col in X.columns:
        signal = X[col].values
        # 用于存放三个统计量
        freq_stats = np.zeros((n_samples, len(stats_labels)))

        # 滑动计算
        for i in range(0, n_samples - window + 1, step):
            seg = signal[i:i+window]
            f, pxx = periodogram(seg, detrend='constant')
            pxx = pxx + 1e-12  # 避免 log(0)
            pxx_norm = pxx / np.sum(pxx)

            freq_stats[i:i+window, 0] = np.mean(pxx)       # 平均功率
            freq_stats[i:i+window, 1] = np.std(pxx)        # 功率标准差
            freq_stats[i:i+window, 2] = -np.sum(pxx_norm * np.log(pxx_norm))  # 谱熵
        # 填充末尾不足一个窗口的部分
        last_start = (n_samples // step) * step
        if last_start < n_samples:
            seg = signal[last_start:]
            if len(seg) >= 4:   # 周期图至少需要4个点
                f, pxx = periodogram(seg, detrend='constant')
                pxx = pxx + 1e-12
                pxx_norm = pxx / np.sum(pxx)
                freq_stats[last_start:, 0] = np.mean(pxx)
                freq_stats[last_start:, 1] = np.std(pxx)
                freq_stats[last_start:, 2] = -np.sum(pxx_norm * np.log(pxx_norm))
            else:
                freq_stats[last_start:, :] = freq_stats[last_start-1, :]  # 复制前一个

        feat_parts.append(freq_stats)
        feat_names += [f"{col}_{stat}" for stat in stats_labels]
        pbar.update(1)


# ======================== 向量化预测工具 ========================
def _compute_errors_vectorized(series, lags, model):
    max_lag = max(lags)
    n = len(series)
    if n <= max_lag:
        return np.zeros(n)

    windowed = sliding_window_view(series, window_shape=max_lag + 1)
    windowed = windowed[:n - max_lag]
    lagged = windowed[:, :max_lag][:, ::-1]
    X_pred = lagged[:, [lag - 1 for lag in lags]]
    preds = model.predict(X_pred)
    errors = np.zeros(n)
    errors[max_lag:] = series[max_lag:] - preds
    return errors


def _make_ar_dataset(series, lags):
    max_lag = max(lags)
    n = len(series)
    if n <= max_lag:
        return np.array([]).reshape(-1, 0), np.array([])

    windowed = sliding_window_view(series, window_shape=max_lag + 1)
    X = windowed[:, :max_lag]
    y = windowed[:, max_lag]
    X = X[:, ::-1]
    X = X[:, [lag - 1 for lag in lags]]
    return X, y


def _train_and_compute_pred_errors(X, y_series, pbar):
    normal_mask = (y_series.values == 0)
    if normal_mask.sum() < 50:
        raise ValueError("Not enough normal samples to train predictors.")

    normal_indices = np.where(normal_mask)[0]
    train_cut = int(len(normal_indices) * PREDICTOR_TRAIN_PROP)
    train_normal_idx = normal_indices[:train_cut]

    X_normal_train = X.iloc[train_normal_idx]
    predictors = {}
    all_errors = np.zeros((len(X), len(FEATURE_COLS)))

    for i, col in enumerate(FEATURE_COLS):
        series = X[col].values
        train_series = X_normal_train[col].values

        X_ar, y_ar = _make_ar_dataset(train_series, PREDICTOR_LAGS)
        if len(X_ar) < 10:
            mean_val = np.mean(train_series)
            all_errors[:, i] = series - mean_val
            predictors[col] = ("mean", mean_val)
            continue

        model = HuberRegressor(epsilon=1.35, max_iter=500)
        model.fit(X_ar, y_ar)
        predictors[col] = model

        all_errors[:, i] = _compute_errors_vectorized(series, PREDICTOR_LAGS, model)

    pbar.update(1)

    error_df = pd.DataFrame(all_errors, columns=[f"{c}_pred_error" for c in FEATURE_COLS])
    error_feats = [error_df.values]
    error_names = list(error_df.columns)

    # 误差 MAD z-score (窗口20)
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

    # 误差滚动标准差 (窗口20)
    for col in error_df.columns:
        roll_std = error_df[col].rolling(window=20, min_periods=1).std().bfill()
        error_feats.append(roll_std.values.reshape(-1, 1))
        error_names.append(f"{col}_errstd20")
        pbar.update(1)

    return predictors, np.hstack(error_feats), error_names


def _compute_pred_errors_inference(X, trained_predictors, pbar):
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

    for col in error_df.columns:
        roll_std = error_df[col].rolling(window=20, min_periods=1).std().bfill()
        error_feats.append(roll_std.values.reshape(-1, 1))
        error_names.append(f"{col}_errstd20")
        pbar.update(1)

    return np.hstack(error_feats), error_names