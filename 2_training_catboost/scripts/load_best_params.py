import optuna
import config
import joblib

def load_best_params(
    study_name=config.study_name,
    storage=config.storage,
    trial_output=config.trial_output,
    best_param_output=config.best_param_output
):
    """
    Load the best hyperparameters from an Optuna study and save the study trials data to a CSV file.
    
    Parameters:
    - study_name (str): The name of the Optuna study to load.
    - storage (str): The storage URL for the Optuna study database.
    - trial_output (str): The path where the study's trials dataframe will be saved as CSV.
    - best_param_output (str): The file path to dump the best hyperparameters using joblib.
    
    Returns:
    - dict: The best hyperparameters found by the Optuna study.
    """
    
    # Load the Optuna study using the provided study name and storage.
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Convert the study trials to a pandas DataFrame and save it as CSV.
    df = study.trials_dataframe()
    df.to_csv(trial_output, index=False)
    
    # Retrieve the best hyperparameters from the study.
    best_params = study.best_params
    
    # Save the best hyperparameters to a file using joblib.
    joblib.dump(best_params, best_param_output)
    
    # Return the best hyperparameters for further use.
    return best_params
