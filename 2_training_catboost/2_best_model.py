# -*- coding: utf-8 -*-
"""
File:    best_model.py
Author:  Kyusik Kim <kyusik.q.kim@gmail.com>
Date:    2025-06-20
Version: 0.1.0

Description
------------
This script trains a CatBoost multi-class classifier using the best hyperparameters
from a prior Optuna study. It:
  1. Logs start time and progress.
  2. Loads and augments the best parameters (e.g. loss_function, eval_metric, grow_policy).
  3. Prepares training and test datasets, including categorical feature handling.
  4. Fits the CatBoostClassifier and persists the trained model.
  5. Evaluates on the test set (accuracy & balanced accuracy) and writes results.

Dependencies
------------
- Python >= 3.12
- config.py         (must define:  
                      • best_param_output  
                      • X_train_path  
                      • X_test_path  
                      • y_train_path  
                      • y_test_path  
                      • load_data_mode  
                      • cat_vars_path  
                    )  
- scripts/          (must expose:  
                      • create_log(msg, path, mode)  
                      • save_best_params()  
                      • data_prep(X_train_path, X_test_path, y_train_path, y_test_path, mode, cat_vars_path)  
                    )

Usage
-----
Edit `config.py` with your file paths, for example:
   ```python
   best_param_output = "optuna/best_params.pkl"
   X_train_path      = "input/X_train.parquet"
   X_test_path       = "input/X_test.parquet"
   y_train_path      = "input/y_train.parquet"
   y_test_path       = "input/y_test.parquet"
   load_data_mode    = "parquet"       # or "csv"
   cat_vars_path     = "config/cat_vars.json"
"""

# ------------------------------------------------------------------------------
# 1. Imports and Setup
# ------------------------------------------------------------------------------
import config
from scripts import *  # Imports helper functions like create_log, data_prep, save_best_params

import datetime  # For recording timestamps in logs
import joblib    # For saving and loading Python objects
from catboost import CatBoostClassifier  # The CatBoost model
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # Metrics for evaluation

# Define the path for logging progress
log_file_path = 'log/model_fit_progress.txt'

# Log the start of the script execution
create_log(f"Code started. Current time is {datetime.datetime.now()}\n", log_file_path, 'w')

# ------------------------------------------------------------------------------
# 2. Load Optuna Best Hyperparameters and Augment Parameters
# ------------------------------------------------------------------------------
# Save the best parameters from a prior Optuna study and load them
save_best_params()
best_params = joblib.load(config.best_param_output)

# Log creation of the new database for finalized Optuna trials
create_log(f"New database is created for finalized optuna trials.\n", log_file_path, 'a')

# Augment the best parameters with additional settings for training the CatBoost model
# Adjust loss function, evaluation metric, tree growth policy, etc.
best_params.update({
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'grow_policy': 'Lossguide',
    'auto_class_weights': 'Balanced',
    'bootstrap_type': 'Bayesian'
})

# ------------------------------------------------------------------------------
# 3. Data Preparation
# ------------------------------------------------------------------------------
# Load training and testing data along with categorical variables.
# Data is expected in 'parquet' format as specified by `mode`.
X_train, X_test, y_train, y_test, cat_vars = data_prep(
    config.X_train_path,
    config.X_test_path,
    config.y_train_path,
    config.y_test_path,
    config.load_data_mode,
    config.cat_vars_path
)

# Log that data import was completed
create_log(f"Data imported. Current time is {datetime.datetime.now()}\n", log_file_path, 'a')

# ------------------------------------------------------------------------------
# 4. Model Training
# ------------------------------------------------------------------------------
# Log the commencement of model training
create_log(f"Fitting the model at {datetime.datetime.now()}\n", log_file_path, 'a')

# Initialize the CatBoostClassifier with the best parameters and categorical features
model = CatBoostClassifier(**best_params, cat_features=cat_vars)

# Train the model using the training data; verbose output every 100 iterations
model.fit(X_train, y_train, verbose=100)

# Save the trained model to disk for later use
joblib.dump(model, 'output/trained_model_catboost.pkl')

# ------------------------------------------------------------------------------
# 5. Evaluation and Logging
# ------------------------------------------------------------------------------
# Make predictions on the test dataset
pred = model.predict(X_test)

# Compute accuracy scores: general accuracy and balanced accuracy
gen_acc = accuracy_score(y_test, pred)
bal_acc = balanced_accuracy_score(y_test, pred)

# Write the evaluation outcomes to an output text file
with open('output/accuracy_outcome.txt', 'w') as f:
    f.write(f'Train data: {len(X_train)}\n')
    f.write(f'General accuracy: {gen_acc}\n')
    f.write(f'Balanced accuracy: {bal_acc}\n')

# Log the completion of the entire process
create_log(f"Whole process has been done at {datetime.datetime.now()}\n", log_file_path, 'a')
