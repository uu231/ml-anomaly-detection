"""Load, validate, and preprocess data."""
import pandas as pd
import numpy as np
from config import TRAIN_PATH, TEST_SIMPLE_PATH, TEST_COMPLEX_PATH, FEATURE_COLS


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
        if anomaly_idx[i] - anomaly_idx[i-1] > 5:
            clusters.append((start, anomaly_idx[i-1]))
            start = anomaly_idx[i]
    clusters.append((start, anomaly_idx[-1]))
    return clusters