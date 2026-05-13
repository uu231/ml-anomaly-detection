# Design: Robust Anomaly Detection in Noisy Time-Series Data

## 1. Problem Overview

Supervised binary anomaly detection on financial market time-series. Given 137,192 time-ordered samples with 33 features and rare anomaly labels (y=1: 0.42%), detect anomalies in two test scenarios:

- **Task 1**: `test_simple.csv` — similar distribution to training
- **Task 2**: `test_complex.csv` — complex, shifted distribution; uses same trained model, no retraining

## 2. Data Characteristics (from EDA)

| Property | Value |
|----------|-------|
| Train samples | 137,192 |
| Features | 33 (f1-f33, all numerical) |
| Positive ratio | 0.42% (570/137,192) |
| Imbalance | 1:240 |
| Missing values | f22-f25: 1 each; f26-f29: 9 each |
| Anomaly location | ALL in last 10% (index ≥ 123,390) |
| Anomaly clustering | 96.8% consecutive pairs (gap=1) |
| Single-feature |r| with y | max 0.0166 — weak |
| Raw+diff LR AUC-PR | 0.20 (linear insufficient) |

**Key insight**: Anomalies are temporally clustered at the tail end, with patterns captured by changes (diffs) rather than absolute values. Non-linear models required.

## 3. Validation Strategy

### Primary: Purged Block Cross-Validation

Identify anomaly clusters (contiguous y=1 segments), then leave-one-cluster-out:

- Each fold validates on different anomaly clusters
- Tests generalization across anomaly sub-patterns
- Most difficult fold: train on late clusters, validate on early clusters (simulates Task 2)

### Secondary: Two-Phase FPR Control

On normal-only regions (first 90% of data), evaluate false positive rate across 4 sampled windows to ensure the model does not over-predict anomalies on normal data.

### Metrics (order of importance)

1. **AUC-PR** — primary for extreme imbalance
2. **F1-score** — balance of precision/recall
3. **Recall** — catching anomalies
4. **Precision** — alarm reliability

**Accuracy is NOT used** — 99.58% baseline by predicting all 0.

## 4. Feature Engineering

For each of the 33 base features, construct:

| Category | Parameters | Rationale |
|----------|-----------|-----------|
| Raw values | f1-f33 | Current state |
| Lag | 1, 2, 3, 5, 10 | Compare current vs past |
| Diff (1st order) | 1, 2, 3 | Rate of change |
| Diff (2nd order) | 1, 2 | Acceleration of change |
| Rolling mean | windows 3, 5, 10, 20, 50 | Local trend |
| Rolling std | windows 3, 5, 10, 20, 50 | Local volatility |
| Rolling min | windows 5, 10, 20 | Recent minimum |
| Rolling max | windows 5, 10, 20 | Recent maximum |
| Rolling z-score | windows 10, 20, 50 | Standardized deviation from history |
| EMA | spans 5, 10, 20 | Exponential trend (recent-weighted) |

**Constraints**:
- All features use ONLY past information (no future leakage)
- Scaler/imputer fit on training split only
- Estimated final dimensionality: ~600-800 features

## 5. Model Architecture

### Primary: LightGBM
- `LGBMClassifier` with `scale_pos_weight` ≈ 240
- Leaf-wise growth, histogram-based
- Handles missing values natively

### Comparison baselines (for report):
- XGBoost (`XGBClassifier` with `scale_pos_weight`)
- CatBoost (if time permits)
- Random Forest (`class_weight="balanced"`)
- Logistic Regression (`class_weight="balanced"`)

### Hyperparameter tuning: Optuna (Bayesian optimization)
- Key params: `num_leaves`, `learning_rate`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- Optimize for validation AUC-PR averaged across folds
- Early stopping with `early_stopping_rounds=50`

## 6. Class Imbalance Handling

1. **scale_pos_weight** ~240 (LightGBM/XGBoost built-in)
2. **Threshold tuning**: Optimize decision threshold on validation (search 0.01-0.99) to maximize F1-score
3. **Evaluation**: AUC-PR as primary metric (insensitive to threshold, robust to imbalance)

## 7. Project Structure

```
ml_project/
├── data/
│   ├── train.csv
│   ├── test_simple.csv
│   └── test_complex.csv
├── src/
│   ├── config.py          # All constants, paths, parameters
│   ├── data_utils.py      # Load, validate, preprocess
│   ├── features.py        # Feature engineering pipeline
│   ├── validation.py      # Purged Block CV, two-phase FPR
│   ├── train.py           # Model training with Optuna
│   ├── evaluate.py        # Metrics, threshold tuning
│   ├── predict.py         # Generate submission files
│   └── utils.py           # Logging, timer, helpers
├── main.py                # Entry point: train/predict modes
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── figures/
└── docs/
    └── superpowers/specs/
```

## 8. Implementation Order

1. `config.py` — constants and paths
2. `data_utils.py` — data loading, validation
3. `features.py` — feature engineering (lag, diff, rolling, EMA, z-score)
4. `validation.py` — Purged Block CV, metrics
5. `train.py` — baseline models, Optuna tuning, final model
6. `evaluate.py` — threshold optimization
7. `predict.py` — generate pred_simple.csv, pred_complex.csv
8. `main.py` — CLI entry point

## 9. Constraints

- Only `train.csv` used for training, validation, and model selection
- Task 2 MUST use the same model, same threshold, same preprocessing as Task 1
- No retraining, fine-tuning, or adaptation for test_complex
- Prediction files: single column `y_pred`, rows match test set exactly
- Temporal order preserved throughout (no random shuffling)
