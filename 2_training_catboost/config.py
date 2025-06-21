import os

# ------------------------------------------------------------------------------
# Database Configuration for Optuna Studies
# ------------------------------------------------------------------------------

# Primary Optuna study database configuration for hyperparameter tuning.
DB_PATH = "optuna_study.db"
storage = f"sqlite:///{DB_PATH}"  # Connection string to access the database.
study_name = 'catboost_optuna'    # Name of the primary Optuna study.

# Finalized study database configuration for saving optimized results.
final_DB_PATH = 'optuna_study_final.db'
final_storage = f'sqlite:///{final_DB_PATH}'  # Connection string for the finalized study.
final_study_name = 'optuna_catboost_final'     # Name of the finalized Optuna study.

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

# File path for logging Optuna related messages.
optuna_log = "log/optuna_log.txt"

# ------------------------------------------------------------------------------
# Data Loading Configuration
# ------------------------------------------------------------------------------

# Mode for loading data files ('csv' or 'parquet').
load_data_mode = 'parquet'

# File paths for training and testing datasets.
X_train_path = "input_data/X_train."  # Base path for training features.
X_test_path = "input_data/X_test."    # Base path for testing features.
y_train_path = "input_data/y_train."   # Base path for training targets.
y_test_path = "input_data/y_test."     # Base path for testing targets.

# JSON file containing a list of categorical variables.
cat_vars_path = 'input_data/variables_list.json'

# ------------------------------------------------------------------------------
# Output File Paths for Optuna Optimization
# ------------------------------------------------------------------------------

# Path to save the CSV file containing trial details.
trial_output = 'output/optuna_trials.csv'

# Path to save the best hyperparameters determined by Optuna.
best_param_output = 'output/best_params.pkl'
