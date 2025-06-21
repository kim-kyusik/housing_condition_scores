# -*- coding: utf-8 -*-
"""
File:    calibration.py
Author:  Kyusik Kim <kyusik.kim@example.com>
Date:    2025-06-20
Version: 0.1.0

Description
-----------
Loads a trained CatBoost model and test data (in Parquet), evaluates performance,
calibrates probabilities with isotonic regression, finds optimal class‐specific
thresholds via F1 maximization, and exports:
  • Classification report to console  
  • Calibrated model (`calibrated_catboost.pkl`)  
  • Best thresholds (`best_threshold.json`)

Dependencies
------------
- Python >= 3.12
- config.py         (must define X_train_path, X_test_path, y_train_path, y_test_path, cat_vars_path)  
- scripts/          (must expose: load_data, load_cat_vars)

Usage
-----
Ensure your working directory is the project root with `config.py` and `scripts/`.  
"""

# Set working directory to the parent directory
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_curve

import config
from scripts import load_data, load_cat_vars


# ——— Helpers ——————————————————————————————————————————————————————————————————

def load_and_reset(path: str, mode: str = "parquet"):
    """Load a dataset and reset its index."""
    df = load_data(path, mode=mode)
    return df.reset_index(drop=True)

def build_feature_labels(test_df: pd.DataFrame):
    """Return a dict mapping feature column → human-readable name."""
    cat_vars = load_cat_vars(config.cat_vars_path)
    cat_labels = [
        'RUCA Classification', 'Basement', 'Construction',
        'Exterior Cover', 'Heating', 'Roof'
    ]
    num_vars = test_df.columns.tolist()[len(cat_vars):]
    num_labels = [
        'Tax Amount', 'Year Built', 'Renovation Year',
        'Parking Spaces', 'Bedrooms', 'Bathrooms',
        'Median Household Income', 'Median Year Built',
        'Median Home Value', 'Lower Home Quartile',
        'Upper Home Quartile', 'Home Owners (%)',
        'Mobile Homes (%)', 'Population Density',
        'Poverty (%)'
    ]
    return dict(zip(cat_vars, cat_labels), **dict(zip(num_vars, num_labels)))

def find_best_thresholds(probs: np.ndarray, y: pd.Series) -> dict:
    """
    For each class, compute the threshold that maximizes F1 on the training set.
    Returns {class_index: threshold}.
    """
    best = {}
    for cls in range(probs.shape[1]):
        y_bin = (y == cls).astype(int)
        p, = probs[:, cls:cls+1],
        prec, rec, thr = precision_recall_curve(y_bin, p)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        best[cls] = thr[np.argmax(f1[:-1])]
    return best

def predict_custom(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    """
    Apply class-specific thresholds: if no prob ≥ its threshold,
    fallback to argmax.
    """
    preds = []
    for p in probs:
        candidates = [i for i, pi in enumerate(p) if pi >= thresholds[i]]
        preds.append(max(candidates, key=lambda i: p[i]) if candidates else p.argmax())
    return np.array(preds)


mode = 'parquet'
X_train = load_and_reset(config.X_train_path, mode)
y_train = load_and_reset(config.y_train_path, mode).squeeze()
X_test  = load_and_reset(config.X_test_path, mode)
y_test  = load_and_reset(config.y_test_path, mode).squeeze()

model = joblib.load('output/trained_model_catboost.pkl')

# 2) Evaluate & print baseline report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

# 3) Calibrate on training data
calib = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
calib.fit(X_train, y_train)
joblib.dump(calib, 'output/calibrated_catboost.pkl')

# 4) Find and save best thresholds
train_probs = calib.predict_proba(X_train)
best_thresh = find_best_thresholds(train_probs, y_train)
with open('output/best_threshold.json', 'w') as f:
    json.dump(best_thresh, f, indent=2)

# 5) Apply thresholds on test set, evaluate again
test_probs = calib.predict_proba(X_test)
y_pred_thresh = predict_custom(test_probs, best_thresh)
print("After calibration & custom thresholds:")
print(classification_report(y_test, y_pred_thresh, digits=4))

