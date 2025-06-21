# save_best_params.py
import optuna
import pandas as pd
import config
import optuna
import joblib

def save_best_params(study_name = config.study_name,
                    storage = config.storage,
                      final_study_name = config.final_study_name,
                    final_storage = config.final_storage,
                    trial_output = config.trial_output,
                    best_param_output = config.best_param_output,
                    n_trials = 50):

    study = optuna.load_study(study_name=study_name, storage=storage)
    
    all_trials = study.get_trials()
    filtered_trials = [trial for trial in all_trials if trial.number <= n_trials]
    
    final_study = optuna.create_study(
        study_name = final_study_name,
        storage=final_storage,
        direction=study.direction, 
        load_if_exists=True)
        
    for trial in filtered_trials:
        final_study.add_trial(trial)

    final_study = optuna.load_study(study_name=final_study_name, storage=final_storage)
    
    # save df
    df = final_study.trials_dataframe()
    df.to_csv(trial_output, index=False)

    best_params = final_study.best_params
    joblib.dump(best_params, best_param_output)

    
