import pandas as pd
import json
from sklearn.utils.class_weight import compute_sample_weight

# ------------------------------------------------------------------------------
# Data Loading and Preprocessing Functions
# ------------------------------------------------------------------------------

def load_data(file_path, mode='csv'):
    """
    Load data from the specified file path using the given mode.
    
    Parameters:
    - file_path (str): The base file path to the data file (excluding the file type extension).
    - mode (str): The format of the file to load ('csv' or 'parquet').
    
    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    # Load as CSV or parquet based on mode
    if mode == 'csv':
        df = pd.read_csv(file_path + mode)
    elif mode == 'parquet': 
        df = pd.read_parquet(file_path + mode, engine='pyarrow')
    return df

def load_cat_vars(cat_vars_path):
    """
    Load a list of categorical variables from a JSON configuration file.
    
    Parameters:
    - cat_vars_path (str): Path to the JSON file containing categorical variable information.
    
    Returns:
    - list: A list of categorical variable names, after removing certain keys.
    """
    # Read the JSON file to get variable information
    with open(cat_vars_path, 'r') as file:
        var_info = json.load(file)
    
    # Extract categorical variables and remove specific entries
    cat_vars = var_info['categorical_vars']
    cat_vars.remove('PROPID')
    cat_vars.remove('CONDITION_CAT')
    cat_vars.remove('FUELTYPE_CAT')
    
    return cat_vars

def make_str(df, cat_vars):
    """
    Convert the specified categorical columns of the DataFrame to string type.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns need conversion.
    - cat_vars (list): The list of categorical column names to convert.
    
    Returns:
    - pd.DataFrame: The updated DataFrame with specified columns as strings.
    """
    # Change each categorical column to string type for consistency
    for cat in cat_vars:
        df[cat] = df[cat].astype(str)
    return df

def make_y(ydf, y_name='CONDITION_encoded'):
    """
    Extract the target column from the given DataFrame.
    
    Parameters:
    - ydf (pd.DataFrame): The DataFrame containing the target variable.
    - y_name (str): The column name of the target variable.
    
    Returns:
    - pd.Series: The extracted target variable.
    """
    return ydf[y_name]

def get_y_weight(y, class_weight='balanced'):
    """
    Compute sample weights for the target variable based on class imbalance.
    
    Parameters:
    - y (pd.Series): The target variable.
    - class_weight (str or dict): The strategy to calculate class weights.
    
    Returns:
    - np.array: An array of sample weights.
    """
    return compute_sample_weight(class_weight=class_weight, y=y)

def make_Xy(X_train, y_train, X_valid, y_valid):
    """
    Combine training and validation sets for features and target variables.
    
    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - X_valid (pd.DataFrame): Validation features.
    - y_valid (pd.Series): Validation target.
    
    Returns:
    - tuple: A tuple containing the combined features and target variables.
    """
    # Concatenate training and validation data for features and target
    X = pd.concat([X_train, X_valid], ignore_index=True).reset_index(drop=True)
    y = pd.concat([y_train, y_valid], ignore_index=True).reset_index(drop=True)
    return X, y

def data_prep(X_train_path, X_test_path, y_train_path, y_test_path, mode, cat_vars_path):
    """
    Prepare and preprocess the data for CatBoost training.
    
    This function loads training and testing feature and target datasets, loads the 
    list of categorical variables, resets the indices, and converts categorical columns 
    to string type.
    
    Parameters:
    - X_train_path (str): File path for training features.
    - X_test_path (str): File path for testing features.
    - y_train_path (str): File path for training target.
    - y_test_path (str): File path for testing target.
    - mode (str): The format of the files ('csv' or 'parquet').
    - cat_vars_path (str): Path to the JSON file with categorical variables info.
    
    Returns:
    - tuple: (X_train, X_test, y_train, y_test, cat_vars)
    """
    # Load data files in the specified mode
    X_train = load_data(X_train_path, mode=mode)
    X_test = load_data(X_test_path, mode=mode)
    y_train = load_data(y_train_path, mode=mode)
    y_test = load_data(y_test_path, mode=mode)

    # Load categorical variables
    cat_vars = load_cat_vars(cat_vars_path)

    # Reset the indices of the DataFrames
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Convert categorical columns to string type
    X_train = make_str(X_train, cat_vars)
    X_test = make_str(X_test, cat_vars)
    
    return X_train, X_test, y_train, y_test, cat_vars
