"""Temporal feature engineering pipeline."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import (
    FEATURE_COLS, LAGS, DIFFS_1, DIFFS_2,
    ROLLING_WINDOWS, ROLLING_STATS, ROLLING_MINMAX_WINDOWS,
    ZSCORE_WINDOWS, EMA_SPANS,
)

TOTAL_STEPS = (
    1 + len(LAGS) + len(DIFFS_1) + len(DIFFS_2)
    + len(ROLLING_WINDOWS) * len(ROLLING_STATS)
    + len(ROLLING_MINMAX_WINDOWS) * 2
    + len(ZSCORE_WINDOWS) + len(EMA_SPANS)
)


def build_features(df, show_progress=True):
    """Build temporal features from raw DataFrame.

    Returns:
        X: feature matrix (numpy array)
        feature_names: list of column names
    """
    X = df[FEATURE_COLS].copy()
    X = X.ffill().fillna(0)

    feature_parts = [X.values]
    feature_names = list(X.columns)

    pbar = tqdm(total=TOTAL_STEPS, desc="  Building features", disable=not show_progress,
                bar_format="{desc}: {n}/{total} |{bar}| {elapsed}")

    # Lag features (ffill: only past information, no future leakage)
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

    # Second-order diff: diff(diff(X, d), d)
    for d in DIFFS_2:
        diff2 = X.diff(d).diff(d).fillna(0)
        diff2.columns = [f"{c}_diff2_{d}" for c in X.columns]
        feature_parts.append(diff2.values)
        feature_names.extend(diff2.columns)
        pbar.update(1)

    # Rolling window statistics
    for w in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            rolled = getattr(X.rolling(window=w, min_periods=1), stat)()
            rolled = rolled.bfill()
            rolled.columns = [f"{c}_roll{stat}_{w}" for c in X.columns]
            feature_parts.append(rolled.values)
            feature_names.extend(rolled.columns)
            pbar.update(1)

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

    # Rolling z-score
    for w in ZSCORE_WINDOWS:
        roll_mean = X.rolling(window=w, min_periods=1).mean().bfill()
        roll_std = X.rolling(window=w, min_periods=1).std().bfill()
        roll_std = roll_std.replace(0, 1e-8)
        zscore = ((X - roll_mean) / roll_std).fillna(0)
        zscore.columns = [f"{c}_zscore_{w}" for c in X.columns]
        feature_parts.append(zscore.values)
        feature_names.extend(zscore.columns)
        pbar.update(1)

    # EMA
    for span in EMA_SPANS:
        ema = X.ewm(span=span, adjust=False, min_periods=1).mean().bfill()
        ema.columns = [f"{c}_ema_{span}" for c in X.columns]
        feature_parts.append(ema.values)
        feature_names.extend(ema.columns)
        pbar.update(1)

    pbar.close()
    X_result = np.hstack(feature_parts)
    return X_result, feature_names


def get_feature_count():
    """Estimate total feature count."""
    n_base = len(FEATURE_COLS)
    n_lag = len(FEATURE_COLS) * len(LAGS)
    n_diff1 = len(FEATURE_COLS) * len(DIFFS_1)
    n_diff2 = len(FEATURE_COLS) * len(DIFFS_2)
    n_roll = len(FEATURE_COLS) * (
        len(ROLLING_WINDOWS) * len(ROLLING_STATS)
        + len(ROLLING_MINMAX_WINDOWS) * 2
    )
    n_zscore = len(FEATURE_COLS) * len(ZSCORE_WINDOWS)
    n_ema = len(FEATURE_COLS) * len(EMA_SPANS)
    return n_base + n_lag + n_diff1 + n_diff2 + n_roll + n_zscore + n_ema
