# -*- coding: utf-8 -*-
"""
File:    optuna.py
Author:  Kyusik Kim <kyusik.q.kim@gmail.com>
Date:    2025-06-20
Version: 0.1.0

Description
------------
This script performs hyperparameter optimization for a CatBoost classification model
using Optuna. It:
  1. Initializes or creates the results database via `db_setup()`.
  2. Loads training features (X) and target (y) from configured file paths.
  3. Loads and casts categorical variable names for CatBoost.
  4. Logs the start and end times of the optimization process to a progress file.
  5. Invokes `run_optuna()` to execute the specified number of trials.

Dependencies
------------
- Python >= 3.12
- config.py         (must define:  
                      • DB_PATH  
                      • X_train_path  
                      • y_train_path  
                      • load_data_mode  
                      • cat_vars_path  
                    )  
- scripts/          (must expose:  
                      • db_setup(db_path)  
                      • load_data(path, mode)  
                      • load_cat_vars(path)  
                      • make_str(df, cat_vars)  
                      • run_optuna(X, y, cat_features, n_trials)  
                    )

Usage
-----
Edit `config.py` to point to your database and data files, for example:
   ```python
   DB_PATH         = "path/to/optuna_results.db"
   X_train_path    = "input/X_train.parquet"
   y_train_path    = "input/y_train.parquet"
   load_data_mode  = "parquet"      # or "csv"
   cat_vars_path   = "config/cat_vars.json"
"""

# Import configuration and helper functions from your project
import config
from scripts import *

# Standard libraries for data handling and logging
import pandas as pd
import numpy as np
import sys
import joblib
import json
import pickle
import logging
from datetime import datetime
import os
import sqlite3

# Machine Learning libraries
import sklearn
from catboost import CatBoostClassifier, Pool  # CatBoost model and Pool class for handling datasets
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split  # Splitting tools for data
from sklearn.metrics import accuracy_score, make_scorer, balanced_accuracy_score  # Metrics for evaluation
from scipy.stats import uniform, truncnorm, randint  # Distributions for hyperparameter sampling

# Optimization library
import optuna

# ------------------------------------------------------------------------------
# 1. Database Setup
# ------------------------------------------------------------------------------
# Create or initialize the database as specified in the config file.
db_setup(config.DB_PATH)

# ------------------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------------------

# Load feature and target data using the helper function from scripts
X = load_data(config.X_train_path, mode=config.load_data_mode)
y = load_data(config.y_train_path, mode=config.load_data_mode)

# Load list of categorical variables
cat_vars = load_cat_vars(config.cat_vars_path)

# Ensure that the index of the data frames is reset (remove old indices)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Convert specified categorical columns to string type for CatBoost compatibility
X = make_str(X, cat_vars)

# ------------------------------------------------------------------------------
# 3. Log Training Progress Start
# ------------------------------------------------------------------------------
# Define the file to log training progress and record the start time.
progress_file = 'log/model_train_progress.txt'
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(progress_file, 'w') as f:
    f.write(f"Optimization begin: {current_time}\n")

# ------------------------------------------------------------------------------
# 4. Run Hyperparameter Optimization
# ------------------------------------------------------------------------------
# Use Optuna to optimize hyperparameters for the CatBoost model.
# The function `run_optuna` is assumed to handle the optimization process.
run_optuna(X, y, cat_features=cat_vars, n_trials=1)

# ------------------------------------------------------------------------------
# 5. Log Training Progress End
# ------------------------------------------------------------------------------
# After optimization, record the completion time in the log file.
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(progress_file, 'a') as f:
    f.write(f"Optimization end: {current_time}\n")
