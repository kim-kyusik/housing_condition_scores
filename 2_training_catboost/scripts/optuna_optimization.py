import optuna
import logging
import config
import numpy as np
import torch

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

def run_optuna(X, y, cat_features, n_trials=50):
    """
    Perform hyperparameter optimization for CatBoost using Optuna with K-fold cross-validation.
    
    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series/DataFrame): Target variable.
    - cat_features (list): List of categorical feature names.
    - n_trials (int): Number of optimization trials to run.
    """
    # Add a file handler to log Optuna messages to the file specified in config.
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(config.optuna_log))
    
    def objective(trial):
        """
        Objective function to optimize.
        Sets up hyperparameters, performs 3-fold cross-validation, and returns the average 
        validation balanced accuracy.
        """
        # Get the number of available GPUs and create a device list.
        n_gpus = torch.cuda.device_count()
        device_list = ','.join(str(i) for i in range(n_gpus))
        
        # Define the hyperparameters to tune.
        params = {
            "iterations": trial.suggest_int('iterations', 5000, 10000),                  # Number of boosting iterations (trees)
            "depth": trial.suggest_int("depth", 10, 200),                                # Depth of each tree
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  # Learning rate (log scale)
            "random_strength": trial.suggest_float("random_strength", 1, 30),            # Randomness level for feature splits
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 4.0),   # Controls bagging temperature
            "bootstrap_type": "Bayesian",                                                # Sampling method for bagging
            "border_count": trial.suggest_int("border_count", 250, 1500),                # Number of splits for numerical features
            "loss_function": "MultiClass",                                               # For softmax probabilities in multiclass classification
            "eval_metric": "MultiClass",                                                 # Evaluation metric for multiclass
            "early_stopping_rounds": 100,
            "grow_policy": "Lossguide",
            "auto_class_weights": "Balanced",                                            # Automatically balance classes
            "verbose": 100
        }
        
        # Set up 3-fold Stratified K-Fold cross-validation to maintain balanced class distribution.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=133)

        train_scores = []
        valid_scores = []

        # Run cross-validation.
        for train_idx, valid_idx in cv.split(X, y):
            # Split the data into training and validation based on the indices.
            X_train, X_valid = X.loc[train_idx, :], X.loc[valid_idx, :]
            y_train, y_valid = y.loc[train_idx, :], y.loc[valid_idx, :]

            # Initialize the CatBoost model using current parameters, categorical features, and GPU settings.
            model = CatBoostClassifier(**params, cat_features=cat_features, task_type='GPU', devices=device_list)
            
            # Train the model with early stopping on the validation set.
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50, verbose=100)
            
            # Make predictions on both training and validation sets.
            train_pred = model.predict(X_train)
            valid_pred = model.predict(X_valid)
    
            # Evaluate the balanced accuracy for both sets.
            train_acc = balanced_accuracy_score(y_train, train_pred)
            valid_acc = balanced_accuracy_score(y_valid, valid_pred)
    
            train_scores.append(train_acc)
            valid_scores.append(valid_acc)

        # Calculate average scores across all folds.
        avg_train = np.mean(train_scores)
        avg_valid = np.mean(valid_scores)
        gap = avg_train - avg_valid  # Measure of overfitting

        # Record metrics as trial attributes for further analysis.
        trial.set_user_attr('avg_train', avg_train)
        trial.set_user_attr('avg_valid', avg_valid)
        trial.set_user_attr('overfitting_gap', gap)
        
        # Return the average validation score for optimization.
        return avg_valid

    # Create or load an existing Optuna study for hyperparameter optimization.
    study = optuna.create_study(
        direction='maximize',
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=True
    )

    # Start the optimization process over the specified number of trials.
    study.optimize(objective, n_trials=n_trials)
