# -*- coding: utf-8 -*-
"""
File:    data_cleaning.py
Author:  Kyusik Kim <kyusik.q.kim@gmail.com>
Date:    2025-06-20
Version: 0.1.0

Description
------------
This script connects to a SQLite database (configured via a separate `config.py`),
computes and exports summary statistics for both numeric and categorical columns,
and applies basic cleaning rules to selected numeric fields by adding “_CLN” columns.

Dependencies
------------
- Python >= 3.12
- config.py    (must define `db` as path to your `.sqlite` file and `table_name` as the target table)

Ensure you have a `report/` directory alongside the script for the CSV outputs.
"""

# Import Packages
import sqlite3
import pandas as pd
from datetime import datetime
import config  # Configuration module containing db and table_name

# Connect to the database and create a cursor.
conn = sqlite3.connect(config.db)
cursor = conn.cursor()
table_name = config.table_name

# Functions for Numeric Data Statistics
def make_statistics(variable, table_name):
    """
    Compute basic statistics (mean, min, max, total rows, non-null and null counts)
    for a given numeric variable from the SQL table.
    
    Parameters:
      variable (str): Column name to compute statistics.
      table_name (str): Name of the table in the database.
    
    Returns:
      DataFrame: DataFrame containing the computed statistics along with the variable name.
    """
    query = f"""
        SELECT AVG({variable}) AS mean,
               MIN({variable}) AS min,
               MAX({variable}) AS max,
               COUNT(*) AS total_rows,
               COUNT({variable}) AS notnull_rows,
               COUNT(*) - COUNT({variable}) AS null_rows
        FROM {table_name};
    """
    df = pd.read_sql_query(query, conn)
    df['variable'] = variable
    return df

# List of numeric variables
num_vars = [
    'LAT', 'LON', 'TAXAMT', 'NUMBLDGS', 'NUMUNITS_NUM', 
    'YEARBUILT', 'RENOYEAR', 'NUMFLOORS_NUM', 'NUMPARKSP', 
    'LIVINGAREA_NUM', 'TOTROOMS', 'BEDROOMS', 'BATHROOMS'
]

# Compute statistics for each numeric variable and store them
stat_list = []
for var in num_vars:
    try:
        stat_list.append(make_statistics(var, table_name))
        print(f"{var} is calculated.")
    except sqlite3.Error as e:
        print(f"Error with {var}: {e} -- variable not in db.")
        continue

# Concatenate and enhance statistics DataFrame
stat_df = pd.concat(stat_list, ignore_index=True)
stat_df['null_pct'] = stat_df['null_rows'] / stat_df['total_rows'] * 100
stat_df = stat_df[['variable'] + [col for col in stat_df.columns if col != 'variable']]
stat_df.to_csv(f'report/statistics_num_var_{datetime.now().date()}.csv', index=False)

# Optionally remove variables with too many nulls.
df_numeric = stat_df[~stat_df['variable'].isin(['NUMUNITS_NUM', 'NUMFLOORS_NUM', 'LIVINGAREA_NUM'])].reset_index(drop=True)

# Function to Read a Single Column from the Database
def read_data(var):
    """
    Reads a column from the SQL table.
    
    Parameters:
      var (str): The column name to read.
    
    Returns:
      DataFrame: The resulting DataFrame with one column named 'value'.
    """
    query = f"SELECT {var} AS value FROM {table_name}"
    return pd.read_sql_query(query, conn)

# Function to Update a Numeric Column in the Table
def update_column(target_var, new_type, condition):
    """
    Alters the table to add a new cleaned column and updates it based on a condition.
    
    Parameters:
      target_var (str): Original column name.
      new_type (str): Data type for the new column (e.g., 'FLOAT', 'INTEGER').
      condition (str): CASE expression conditions to clean the column.
    """
    new_col = f"{target_var}_CLN"
    query_alter = f"ALTER TABLE {table_name} ADD COLUMN {new_col} {new_type};"
    cursor.execute(query_alter)
    conn.commit()
    
    query_update = f"""
        UPDATE {table_name}
        SET {new_col} = CASE
            {condition}
        END;
    """
    cursor.execute(query_update)
    conn.commit()
    print(f"Updated {target_var} to new column {new_col}.")


# Update Numeric Columns Using a Generic Function
update_column("TAXAMT", "FLOAT", "WHEN TAXAMT < 10 OR TAXAMT > 50000 THEN NULL ELSE TAXAMT")
update_column("YEARBUILT", "INTEGER", "WHEN YEARBUILT < 1600 OR YEARBUILT > 2024 THEN NULL ELSE YEARBUILT")
update_column("RENOYEAR", "INTEGER", "WHEN RENOYEAR < 1800 OR RENOYEAR > 2024 THEN NULL ELSE RENOYEAR")
update_column("NUMPARKSP", "INTEGER", "WHEN NUMPARKSP > 10 THEN NULL ELSE NUMPARKSP")
update_column("BEDROOMS", "FLOAT", "WHEN BEDROOMS > 30 OR BEDROOMS < 0 THEN NULL ELSE BEDROOMS")
update_column("BATHROOMS", "FLOAT", "WHEN BATHROOMS > 20 THEN NULL ELSE BATHROOMS")


# Functions for Categorical Data Statistics
def make_statistics_cat(variable, table_name):
    """
    Computes frequency counts for a categorical variable from the SQL table.
    
    Parameters:
      variable (str): The categorical column name.
      table_name (str): The table name.
    
    Returns:
      DataFrame: A DataFrame with counts (frequency) and the variable name.
    """
    query = f"""
        SELECT {variable} AS item, COUNT(*) AS freq
        FROM {table_name}
        GROUP BY {variable};
    """
    df = pd.read_sql_query(query, conn)
    df['variable'] = variable
    return df

# List of categorical variables
cat_vars = [
    'BASMNTTYPE_CAT', 'CONDITION_CAT', 'CONSTTYPE_CAT', 
    'EXTCOVER_CAT', 'FUELTYPE_CAT', 'HEATTYPE_CAT', 'ROOFTYPE_CAT'
]

cat_stat_list = []
for var in cat_vars:
    try:
        cat_stat_list.append(make_statistics_cat(var, table_name))
        print(f"{var} is completed.")
    except sqlite3.OperationalError as e:
        if "no such column" in str(e):
            print(f"{var} not found in the table.")
            continue

stat_cat_df = pd.concat(cat_stat_list, ignore_index=True)
stat_cat_df = stat_cat_df[['variable'] + [col for col in stat_cat_df.columns if col != 'variable']]
# Calculate percentage frequency for the first variable group (assumes uniform total, adjust if needed)
total_rows = stat_cat_df.groupby('variable')['freq'].sum().iloc[0]
stat_cat_df['pct'] = stat_cat_df['freq'] / total_rows * 100
stat_cat_df.to_csv(f'report/statistics_num_cat_{datetime.now().date()}.csv', index=False)
