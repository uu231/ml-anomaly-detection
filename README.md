# Robust Anomaly Detection in Noisy Time-Series Data

Supervised anomaly detection on financial market time-series. 137k samples, 33 features, **1:240 extreme class imbalance**, anomalies clustered at the tail end of the sequence.

## Quick Start

```bash
# Train (generates outputs/models/model.pkl)
conda activate ml_env && python -u main.py --mode train

# Predict (generates pred_simple.csv, pred_complex.csv)
conda activate ml_env && python -u main.py --mode predict

# Both
conda activate ml_env && python -u main.py --mode all
```

## Results

| Metric | CV (Purged Block) | Final Model |
|--------|:--:|:--:|
| AUC-PR | 0.9193 ± 0.07 | 0.9953 |
| F1 | 0.5426 | 0.9840 |
| Recall | 47.2% | 97.4% |
| Precision | 70.2% | 99.5% |
| Threshold | 0.50 | 0.14 |
| FPR (normal region) | — | 0.00% |

| Test Set | Samples | Predicted Positives |
|----------|---------|:--:|
| test_simple | 25,647 | 607 (2.37%) |
| test_complex | 34,542 | 494 (1.43%) |

## Pipeline

```
train.csv → Feature Engineering → StandardScaler → LightGBM → Threshold Tuning → Predict
                ├─ lag(1,2,3,5,10)
                ├─ diff1(1,2,3)
                ├─ diff2(1,2)
                ├─ rolling mean/std (w=3,5,10,20,50)
                ├─ rolling min/max (w=5,10,20)
                ├─ rolling z-score (w=10,20,50)
                └─ EMA (span=5,10,20)
```

## Design

- **Validation**: Purged Block CV (leave-one-anomaly-cluster-out) + Two-phase FPR control on normal-only data
- **Model**: LightGBM with `scale_pos_weight`, early stopping, Leaf-wise growth
- **Imbalance**: `scale_pos_weight=240` + threshold tuning maxing F1
- **Reproducibility**: fixed seed (42), complete pipeline saved in `model.pkl`

## Requirements

```
lightgbm  xgboost  optuna  scikit-learn  pandas  numpy  joblib  tqdm
```

## Submission Files

- `outputs/predictions/pred_simple.csv` — Task 1 predictions
- `outputs/predictions/pred_complex.csv` — Task 2 predictions
- `outputs/models/model.pkl` — Trained pipeline (model + scaler + threshold)
- `src/` + `main.py` — Complete source code
