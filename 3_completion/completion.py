def cleaning_data(df, numeric_feature_names):
    """
    Cleans the input DataFrame by:
      - Converting specified numeric features to float.
      - Mapping RUCA2010 numeric codes to descriptive text,
        and handling a specific missing code (99).
      - Replacing non-positive values in 'med_yr_built' with None.
      - Standardizing categorical variable names for several columns.

    Parameters:
      df (pandas.DataFrame): The DataFrame to clean.
      numeric_feature_names (list of str): List of column names that should be numeric.
    
    Returns:
      pandas.DataFrame: The cleaned DataFrame with reset index.
    """
    # Convert numeric columns to float.
    for num in numeric_feature_names:
        df[num] = df[num].astype('float')
    
    # Set RUCA2010 value of 99 (used for missing) to None.
    df.loc[df['RUCA2010'] == 99, 'RUCA2010'] = None
    
    # Define a mapping for RUCA2010 values.
    ruca_dict = {
        1: 'Urban',
        2: 'Suburban',
        3: 'Suburban',
        4: 'Rural',
        5: 'Rural',
        6: 'Rural',
        7: 'Rural',
        8: 'Rural',
        9: 'Rural',
        10: 'Rural'
    }
    # Map RUCA2010 numeric values to their corresponding category names.
    df['RUCA2010'] = df['RUCA2010'].map(ruca_dict)
    
    # Replace non-positive values in 'med_yr_built' with None.
    df.loc[df['med_yr_built'] <= 0, 'med_yr_built'] = None

    # Standardize categorical entries for the BASMNTTYPE_CAT column.
    col = 'BASMNTTYPE_CAT'
    df.loc[df[col] == 'Improved Basement', col] = 'Improved'
    df.loc[df[col] == 'Unspecified Basement', col] = 'Unspecified'
    df.loc[df[col] == 'Unfinished Basement', col] = 'Unfinished'
    df.loc[df[col] == 'No Basement', col] = 'No'
    df.loc[df[col] == 'Partial Basement', col] = 'Partial'
    df.loc[df[col] == 'Full Basement', col] = 'Full'
    df.loc[df[col] == 'Daylight, Full', col] = 'FullDaylight'
    df.loc[df[col] == 'Daylight, Partial', col] = 'PartialDaylight'

    # Standardize categorical entries for the EXTCOVER_CAT column.
    col = 'EXTCOVER_CAT'
    df.loc[df[col] == 'Siding (Alum/Vinyl)', col] = 'Siding'
    df.loc[df[col] == 'Siding Not (aluminum, vinyl, etc.) ', col] = 'SidingOthers'
    df.loc[df[col] == 'Tilt-up (pre-cast concrete)', col] = 'Tilt_up'
    df.loc[df[col] == 'Shingle (Not Wood)', col] = 'Shingle'

    # Standardize categorical entries for the HEATTYPE_CAT column.
    col = 'HEATTYPE_CAT'
    df.loc[df[col] == 'Forced air unit', col] = 'ForcedAirUnit'
    df.loc[df[col] == 'Wood Burning', col] = 'WoodBurning'

    # Reset index to ensure a clean DataFrame.
    df = df.reset_index(drop=True)

    return df


def create_prediction_df(df, feature_names):
    """
    Cleans the data and creates a prediction DataFrame.
    
    This function performs the following steps:
      1. Cleans the DataFrame using 'cleaning_data'. The column names used for cleaning 
         are assumed to be in the global variable 'feature_names' starting from index 6.
      2. Selects a set of columns, including 'PROPID', 'CONDITION_CAT', and all columns in 'feature_names'.
      3. Constructs a prediction DataFrame from the columns in 'feature_names'.
    
    Parameters:
      df (pandas.DataFrame): The original data.
      feature_names: list of features
    
    Returns:
      tuple: (cleaned dataframe with all columns, dataframe prepared for prediction)
    """

    # Clean the dataframe using the cleaning_data function.
    df = cleaning_data(df, numeric_feature_names=feature_names[6:])
    # Filter for specific columns: 'PROPID', 'CONDITION_CAT', and all features in feature_names.
    col_filter = ['PROPID', 'CONDITION_CAT', 'LAT', 'LON'] + feature_names
    df = df.loc[:, col_filter]
    # Extract the features portion that will be used for model prediction.
    df_prediction = df.loc[:, feature_names]

    return df, df_prediction


def create_data_predict(data):
    """
    Prepares the data for prediction by:
      - Replacing None values with np.nan.
      - Converting categorical variables to string type.

    Parameters:
      data (pandas.DataFrame): The input DataFrame that may contain missing values as None
        and categorical variables.
    
    Returns:
      pandas.DataFrame: The DataFrame with None replaced by np.nan and 
                        categorical variables cast as strings.
    """
    import numpy as np  # ensure numpy is imported here if not already
    
    # Replace Python None with np.nan (useful for consistency in missing data representation).
    data = data.replace({None: np.nan})
    
    # Specify the categorical variables that need to be cast as strings.
    cat_vars = [
        'RUCA2010', 
        'BASMNTTYPE_CAT', 
        'CONSTTYPE_CAT', 
        'EXTCOVER_CAT', 
        'HEATTYPE_CAT',
        'ROOFTYPE_CAT'
    ]
    # Convert categorical variables to string type.
    for cat in cat_vars:
        data.loc[:, cat] = data[cat].astype(str)

    return data


def create_final_df(df, df_prediction, feature_names, model, calibration=False):
    """
    Creates the final DataFrame with condition predictions and numeric codes.

    This function predicts property condition using a given model and a prediction
    DataFrame (df_prediction), then combines these predictions with existing condition
    labels in the original DataFrame (df). It maps numeric condition scores to descriptive
    labels and vice versa, and returns a DataFrame containing specific columns such as 
    property ID, the complete condition (either from data or prediction), and the numeric 
    condition code, along with all feature columns defined in the global variable 'feature_names'.

    Parameters:
      df (pandas.DataFrame): The original DataFrame, which includes at least the following 
                             columns: 'PROPID', 'CONDITION_CAT', and all features in 'feature_names'.
      df_prediction (pandas.DataFrame): The DataFrame containing feature values for making predictions.
      feature_names: list of features
      model: The trained model that supports a predict() method and returns predictions as arrays.

    Returns:
      pandas.DataFrame: A DataFrame with columns 'PROPID', 'CONDITION_COMPLETE', 
                        'CONDITION_NUM', plus all columns listed in 'feature_names'.
    """

    # Create a mapping from numeric condition codes to descriptive condition labels.
    condition = {
        0: 'Unsound',
        1: 'Poor',
        2: 'Fair',
        3: 'Average',
        4: 'Good',
        5: 'Excellent'
    }

    # Reverse the condition mapping to convert descriptive labels back to numeric codes.
    rev_condition = {v: k for k, v in condition.items()}

    # Use the model to predict conditions using the df_prediction DataFrame.
    # Note: 'res' is expected to be an array of predictions where each prediction is like [x]
    res = model.predict(df_prediction)
    # Convert the numeric prediction into a descriptive label using the 'condition' dictionary.
    if calibration == True:
      res = [condition[x] for x in res]
    else: 
      res = [condition[x[0]] for x in res]
    # Add the predicted condition labels to the DataFrame under the column 'CONDITION_PRED'.
    df['CONDITION_PRED'] = res

    # Create a complete condition column:
    # Use the existing 'CONDITION_CAT' column if available; otherwise, fall back to the predicted value.
    df['CONDITION_COMPLETE'] = df['CONDITION_CAT'].fillna(df['CONDITION_PRED'])

    # Map the complete condition back to its numeric code using the reversed mapping.
    df['CONDITION_NUM'] = df['CONDITION_COMPLETE'].map(rev_condition)
    
    # Define the final set of columns to be included.
    # Assumes that 'feature_names' is a global variable containing the names of feature columns.
    final_col_filter = ['PROPID', 'CONDITION_CAT', 'CONDITION_COMPLETE', 'CONDITION_NUM', 'LAT', 'LON'] + feature_names
    # Subset the DataFrame to include only the desired columns.
    df = df.loc[:, final_col_filter]

    return df

