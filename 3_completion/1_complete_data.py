# -*- coding: utf-8 -*-
"""
File:    complete_data.py
Author:  Kyusik Kim <kyusik.kim@example.com>
Date:    2025-06-20
Version: 0.1.0

Description
-----------
This script reads property records in chunks from a SQLite database, applies a trained
CatBoost model (both raw and calibrated versions) to generate condition predictions,
and writes two tables into a DuckDB database:
  1. A full results table (`full_completion` and `full_completion_calib`) containing all
     original columns plus the model’s predicted scores and labels.
  2. A summary table (`condition_score` and `condition_score_calib`) with only the key
     identifier (`PROPID`), categorical and numeric condition outputs, and coordinates.

It automatically clears any existing SQLite “completed_data.sqlite” file, then processes
the source table twice—first with the raw model, then with the calibrated model—appending
results to DuckDB for efficient downstream queries.

Dependencies
------------
- Python >= 3.12
- completion.py     (project helper module defining `create_prediction_df`,  
                        `create_data_predict`, `create_final_df`)  
- config.py         (project configuration module)

Usage
-----
Configure your environment and paths at the top of the script:
   SQLITE_DB = "path/Data_Warren_12_23_2024.sqlite"
   DUCKDB_URI = "path/completed_data.duckdb"
   SOURCE_TABLE = "source_table"
   CHUNK_SIZE = 1_000_000
   RAW_MODEL_PATH = "path/output/trained_model_catboost.pkl"
   CALIB_MODEL_PATH = "path/output/calibrated_catboost.pkl"
   SQLITE_COMPLETE_DB = "completed_data.sqlite"
"""


from completion import * 
import config

import os
import sqlite3
import pandas as pd
import joblib
from sqlalchemy import create_engine
from completion import create_prediction_df, create_data_predict, create_final_df

# — Constants ————————————————————————————————————————————————————————————
SQLITE_DB = "sqlite_db/Real_Estate_Property_Data_Warren_12_23_2024.sqlite"
DUCKDB_URI = "duckdb:///sqlite_db///completed_data.duckdb"
SOURCE_TABLE = "data_final_v25.02.02"
CHUNK_SIZE = 1_000_000
RAW_MODEL_PATH = "proj_catboost/output/trained_model_catboost.pkl"
CALIB_MODEL_PATH = "proj_catboost/output/calibrated_catboost.pkl"
SQLITE_COMPLETE_DB = "completed_data.sqlite"

# — Setup connections ——————————————————————————————————————————————————————
sqlite_conn = sqlite3.connect(SQLITE_DB)
duck_engine = create_engine(DUCKDB_URI)

def clear_sqlite_db(path: str):
    """Delete existing SQLite DB if present."""
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted database file: {path}")

def run_completion(model_path: str, calibrated: bool):
    """
    Process SOURCE_TABLE in chunks, apply model, and write
    both full and selected tables to DuckDB.
    """
    model = joblib.load(model_path)
    feature_names = (
        model.calibrated_classifiers_[0].estimator.feature_names_
        if calibrated
        else model.feature_names_
    )
    suffix = "_calib" if calibrated else ""
    full_tbl = f"full_completion{suffix}"
    score_tbl = f"condition_score{suffix}"
    query = f'SELECT * FROM "{SOURCE_TABLE}";'
    processed = 0

    for chunk in pd.read_sql_query(query, sqlite_conn, chunksize=CHUNK_SIZE):
        df, preds = create_prediction_df(chunk, feature_names)
        preds = create_data_predict(preds)
        df = create_final_df(df, preds, feature_names, model, calibration=calibrated)

        df.to_sql(full_tbl, duck_engine, if_exists="append", index=False)
        df[["PROPID","CONDITION_CAT","CONDITION_COMPLETE","CONDITION_NUM","LAT","LON"]].to_sql(score_tbl, duck_engine, if_exists="append", index=False)

        processed += len(chunk)
        print(f"Appended {processed} rows to {full_tbl}, {score_tbl}")

clear_sqlite_db(SQLITE_COMPLETE_DB)
run_completion(RAW_MODEL_PATH, calibrated=False)
run_completion(CALIB_MODEL_PATH, calibrated=True)