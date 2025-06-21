# -*- coding: utf-8 -*-
"""
File:    concat_data.py
Author:  Kyusik Kim <kyusik.q.kim@gmail.com>
Date:    2025-06-20
Version: 0.1.0

Description
------------
This script orchestrates a multi‐step data processing pipeline against a SQLite database
(configured via `config.py`) and external geospatial and socioeconomic data.
The final outcome will be produced as .parquet and .csv files.

Dependencies
------------
- Python >= 3.12
- config.py         (must define:  
                      • db  
                      • table_name  
                      • ruca_file_loc  
                      • tract_census_file  
                      • bg_census_file  
                      • ruca_table  
                      • ACS_bg_table  
                      • ACS_tract_table  
                      • final_data_table_name  
                      • var_list_file  
                      • csv_filename  
                      • final_data_export_folder  
                      • criteria_null_count  
                    )

Usage
-----
1. Configure `config.py` with your paths and table names.  
2. Ensure the output directories exist:  
   - `input_data_parquet/`  
   - `input_data_csv/`  
"""


# Import Packages
import sqlite3
import pandas as pd
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import config  # Assumes config.py contains database, table names, and file paths

# Basic DB Connection and Table Settings
conn = sqlite3.connect(config.db)
cursor = conn.cursor()

# The main table from which we start, and external table names.
main_table = config.table_name


# Helper Function: Process and Append Chunks to a Table
def process_and_append(query, new_table, chunk_size, process_chunk_func):
    """
    Reads data in chunks from the database using a given query, processes
    each chunk with process_chunk_func, and appends the result to a new table.
    
    Parameters:
        query (str): SQL query to select data.
        new_table (str): Name of the table where processed data will be appended.
        chunk_size (int): Number of rows per chunk.
        process_chunk_func (function): A function that takes a DataFrame and returns a processed DataFrame.
    """
    chunk_running = 0
    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
        processed_chunk = process_chunk_func(chunk)
        chunk_running += len(chunk)  # Use actual length for progress reporting
        print(f"Appending chunk; total rows processed: {chunk_running}")
        processed_chunk.to_sql(new_table, conn, if_exists='append', index=False)

# ---------------------------
# Section 1: Join External RUCA Data with Main Table
# ---------------------------
# Load geography (census tract) data (GeoPackage)
geo_data = gpd.read_file(config.tract_census_file)

# Import RUCA CSV, ensure GEOID is string, and keep only relevant columns.
ruca = pd.read_csv(config.ruca_file_loc, dtype={'GEOID': str})
ruca['GEOID10'] = ruca['GEOID']
ruca = ruca[['GEOID10', 'RUCA2010']]

# Define query: select property id and coordinates from the main table.
query_main = f"SELECT PROPID, LAT, LON FROM {main_table};"
new_table_ruca = config.ruca_table

def process_ruca_chunk(chunk):
    # Create Point geometry from LON, LAT and convert DataFrame to GeoDataFrame.
    chunk['geometry'] = [Point(lon, lat) for lon, lat in zip(chunk['LON'], chunk['LAT'])]
    gdf = gpd.GeoDataFrame(chunk, geometry='geometry', crs="EPSG:4326")
    
    # Reproject if necessary
    if gdf.crs != geo_data.crs:
        gdf = gdf.to_crs(geo_data.crs)
    
    # Spatial join with geo_data (census tract info)
    joined = gpd.sjoin(gdf, geo_data, how='left', predicate='intersects')
    # Select needed columns from join.
    cols_keep = ['PROPID', 'STATEFP10', 'COUNTYFP10', 'TRACTCE10', 'GEOID10']
    joined = joined[cols_keep]
    # Merge (join) with RUCA table based on GEOID10.
    return pd.merge(joined, ruca, how='left', on='GEOID10')

# Process RUCA join in chunks.
process_and_append(query_main, new_table_ruca, chunk_size=100000, process_chunk_func=process_ruca_chunk)

# ---------------------------
# Section 2: Add ACS Information at the Block Group Level
# ---------------------------
# Load ACS Block Group geographic data.
geo_bg = gpd.read_file(config.bg_census_file)
# Rename the GEOID column to GEOID12 and drop non-essential columns.
geo_bg = geo_bg.rename(columns={'GEOID': 'GEOID12'}).drop(columns=['state', 'county', 'tract', 'block group'])
# List of ACS block group columns to be stored (excluding geometry).
acs_bg_cols = list(geo_bg.columns)
acs_bg_cols.remove('geometry')

new_table_acs_bg = config.ACS_bg_table

def process_acs_bg_chunk(chunk):
    # Create geometry column from LON, LAT.
    chunk['geometry'] = [Point(lon, lat) for lon, lat in zip(chunk['LON'], chunk['LAT'])]
    gdf = gpd.GeoDataFrame(chunk, geometry='geometry', crs="EPSG:4326")
    if gdf.crs != geo_bg.crs:
        gdf = gdf.to_crs(geo_bg.crs)
    # Spatial join with ACS block group data.
    joined = gpd.sjoin(gdf, geo_bg, how='left', predicate='intersects')
    # Select property ID and ACS columns.
    cols_keep = ['PROPID'] + acs_bg_cols
    return joined[cols_keep]

process_and_append(query_main, new_table_acs_bg, chunk_size=1000000, process_chunk_func=process_acs_bg_chunk)

# ---------------------------
# Section 3: Add ACS Information at the Tract Level
# ---------------------------
# Load ACS Tract data and rename GEOID to GEOID11.
geo_tract = gpd.read_file(config.tract_census_file)
geo_tract = geo_tract.rename(columns={'GEOID': 'GEOID11'})
acs_tract_cols = list(geo_tract.columns)
acs_tract_cols.remove('geometry')

new_table_acs_tract = config.ACS_tract_table

def process_acs_tract_chunk(chunk):
    # Create geometry from coordinates.
    chunk['geometry'] = [Point(lon, lat) for lon, lat in zip(chunk['LON'], chunk['LAT'])]
    gdf = gpd.GeoDataFrame(chunk, geometry='geometry', crs="EPSG:4326")
    if gdf.crs != geo_tract.crs:
        gdf = gdf.to_crs(geo_tract.crs)
    # Perform spatial join with tract data.
    joined = gpd.sjoin(gdf, geo_tract, how='left', predicate='intersects')
    # Keep only selected columns.
    cols_keep = ['PROPID', 'GEOID11', 'poverty_pct']
    return joined[cols_keep]

process_and_append(query_main, new_table_acs_tract, chunk_size=1000000, process_chunk_func=process_acs_tract_chunk)

# ---------------------------
# Section 4: Generate and Export Final Data & Variable List
# ---------------------------
# Join the main table with the external variables via SQL JOINs.
final_query = f"""
    SELECT * 
    FROM {main_table}
    LEFT JOIN {new_table_ruca} ON {main_table}.PROPID = {new_table_ruca}.PROPID
    LEFT JOIN {new_table_acs_bg} ON {main_table}.PROPID = {new_table_acs_bg}.PROPID
    LEFT JOIN {new_table_acs_tract} ON {main_table}.PROPID = {new_table_acs_tract}.PROPID;
"""
final_table = config.final_data_table_name

# Process final data in chunks and remove duplicated columns.
chunk_running = 0
chunk_size = 100000
for chunk in pd.read_sql_query(final_query, conn, chunksize=chunk_size):
    chunk_running += len(chunk)
    print(f"Appending final data: {chunk_running} rows processed.")
    # Remove duplicated columns (e.g., duplicate PROPID columns)
    chunk = chunk.loc[:, ~chunk.columns.duplicated()]
    chunk.to_sql(final_table, conn, if_exists='append', index=False)

# Export variable lists based on column name patterns.
# Load final table into a DataFrame to extract column names.
df_final = pd.read_sql_query(f"SELECT * FROM {final_table};", conn)
all_cols = list(df_final.columns)

# Identify categorical columns by '_CAT' in the name and cleaned numeric columns by '_CLN'
cols_cat = [col for col in all_cols if '_CAT' in col]
cols_cln = [col for col in all_cols if '_CLN' in col]
# Define additional ACS-related numeric columns.
cols_acs = ['med_hh_inc', 'med_yr_built', 'med_home_val', 'low_home_quart', 'upp_home_quart', 'home_owner_pct', 
            'mobile_home_pct', 'pop_dense', 'poverty_pct']

cols_categorical = ['PROPID', 'RUCA2010'] + cols_cat
cols_numeric = cols_cln + cols_acs

# Build and export the variable mapping as JSON.
var_mapping = {'categorical_vars': cols_categorical, 'numeric_vars': cols_numeric}
with open(config.var_list_file, 'w') as file:
    json.dump(var_mapping, file, indent=4)

# Create a combined list of columns for training data export.
export_cols = var_mapping['categorical_vars'] + var_mapping['numeric_vars']


# ---------------------------
# Section 5: Export Training Data Set CSV
# ---------------------------
# Query: select only records with non-null CONDITION_CAT.
query_train = f"""
    SELECT {', '.join(export_cols)}
    FROM "{final_table}"
    WHERE CONDITION_CAT IS NOT NULL;
"""

csv_filename = config.csv_filename
chunk_size = 1000000
first_chunk = True  # Flag to write header only once
chunk_running = 0

for chunk in pd.read_sql_query(query_train, conn, chunksize=chunk_size):
    chunk.to_csv(csv_filename, mode='a', header=first_chunk, index=False)
    first_chunk = False
    chunk_running += len(chunk)
    print(f"Written to CSV: {chunk_running} rows processed.")


# ---------------------------
# Section 6: Split the Training Data Set into Training and Testing Data Sets for HPC
# ---------------------------

with open(r'raw_data/variables_list.json', 'r') as file:
    var_info = json.load(file)

# cat columns
cat_vars = var_info['categorical_vars']
cat_vars.remove('FUELTYPE_CAT')
num_vars = var_info['numeric_vars']

df = pd.read_csv(csv_filename)
totaldf = len(df) # Number of original data

df.loc[df['RUCA2010'] == 99, 'RUCA2010'] = None
ruca_dict = {
    1:'Urban',
    2:'Suburban',
    3:'Suburban',
    4:'Rural',
    5:'Rural',
    6:'Rural',
    7:'Rural',
    8:'Rural',
    9:'Rural',
    10:'Rural'
}
df['RUCA2010'] = df['RUCA2010'].map(ruca_dict)
df.loc[df['med_yr_built'] <= 0, 'med_yr_built'] = None

# replace variable
cols = df.columns

col = cols[3]
df.loc[df[col] == 'Improved Basement', col] = 'Improved'
df.loc[df[col] == 'Unspecified Basement', col] = 'Unspecified'
df.loc[df[col] == 'Unfinished Basement', col] = 'Unfinished'
df.loc[df[col] == 'No Basement', col] = 'No'
df.loc[df[col] == 'Partial Basement', col] = 'Partial'
df.loc[df[col] == 'Full Basement', col] = 'Full'
df.loc[df[col] == 'Daylight, Full', col] = 'FullDaylight'
df.loc[df[col] == 'Daylight, Partial', col] = 'PartialDaylight'

col = cols[5]
df.loc[df[col] == 'Siding (Alum/Vinyl)', col] = 'Siding'
df.loc[df[col] == 'Siding Not (aluminum, vinyl, etc.) ', col] = 'SidingOthers'
df.loc[df[col] == 'Tilt-up (pre-cast concrete)', col] = 'Tilt_up'
df.loc[df[col] == 'Shingle (Not Wood)', col] = 'Shingle'

col = cols[7]
df.loc[df[col] == 'Forced air unit', col] = 'ForcedAirUnit'
df.loc[df[col] == 'Wood Burning', col] = 'WoodBurning'

# remove FUELTYPE_CAT
df.drop(columns=['FUELTYPE_CAT'], inplace=True)
df = df.dropna(subset=df.select_dtypes(include=[np.number]).columns).reset_index(drop=True)

# Remove na columns based on the count
df["na_col"] = df.isnull().sum(axis=1)

df_reduced = df[df.na_col <= criteria_null_count] # remove rows with more than 4 null
df_reduced = df_reduced.drop(['na_col'], axis=1)

# split data
df = df_reduced.copy()
df.loc[df['CONDITION_CAT'] == "Excellent", "CONDITION_encoded"] = 5
df.loc[df['CONDITION_CAT'] == "Good", "CONDITION_encoded"] = 4
df.loc[df['CONDITION_CAT'] == "Average", "CONDITION_encoded"] = 3
df.loc[df['CONDITION_CAT'] == "Fair", "CONDITION_encoded"] = 2
df.loc[df['CONDITION_CAT'] == "Poor", "CONDITION_encoded"] = 1
df.loc[df['CONDITION_CAT'] == "Unsound", "CONDITION_encoded"] = 0

# Get 80% rows randomly
df = df.sample(frac=0.8, random_state=133)

for cat in cat_vars:
    df[cat] = df[cat].astype('category')

for num in num_vars:
    df[num] = df[num].astype('float')

df[cat_vars] = df[cat_vars].astype(str)

# Drop PROPID
df = df.drop(['PROPID', 'CONDITION_CAT'], axis=1)

# Convert large float and integer columns to smaller types
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

y_col_name = "CONDITION_encoded"
y = df[y_col_name]
X = df.drop(y_col_name, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=133)

# Reset index
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = pd.DataFrame(y_train).reset_index(drop=True)
y_test = pd.DataFrame(y_test).reset_index(drop=True)

# Export parquet
X_train.to_parquet("input_data_parquet/X_train.parquet", engine='pyarrow')
X_test.to_parquet("input_data_parquet/X_test.parquet", engine='pyarrow')

y_train.to_parquet("input_data_parquet/y_train.parquet", engine='pyarrow')
y_test.to_parquet("input_data_parquet/y_test.parquet", engine='pyarrow')

# Export CSV
X_train.to_csv("input_data_csv/X_train.csv", index=False)
X_test.to_csv("input_data_csv/X_test.csv", index=False)

y_train.to_csv("input_data_csv/y_train.csv", index=False)
y_test.to_csv("input_data_csv/y_test.csv", index=False)

# Create txt file
reducedf = len(df_reduced)
pctdf = len(df_reduced)/totaldf*100
n_X_train = len(X_train)
n_X_test = len(X_test)
n_y_train = len(y_train)
n_y_test = len(y_test)

with open("input_data_parquet/data_reduction.txt", "w") as f:
    f.write(f"This separation is based on 80% random selection with 8:2 split for train and testing dataset.\n")
    f.write(f"Original total data: {totaldf}\n")
    f.write(f"Reduced data: {reducedf}\n")
    f.write(f"Percent of reduced data: {pctdf:.2f}\n")
    f.write(f"Number of null reduced: {criteria_null_count}\n")
    f.write(f"----------------------------------------------\n")
    f.write(f"Number of X_train: {n_X_train}\n")
    f.write(f"Number of X_test: {n_X_test}\n")
    f.write(f"Number of y_train: {n_y_train}\n")
    f.write(f"Number of y_test: {n_y_test}\n")

print(f"All procedure has been done.")