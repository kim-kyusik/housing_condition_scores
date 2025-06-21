import os
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------------------------
# Set Working Directory
# ------------------------------------------------------------------------------
# Define the working directory path and change the current working directory.
wd = ""
os.chdir(wd)

# ------------------------------------------------------------------------------
# Define Housing Condition Mappings
# ------------------------------------------------------------------------------
# A dictionary mapping integer keys to descriptive condition labels.
condition = {
    0: 'Unsound',
    1: 'Poor',
    2: 'Fair',
    3: 'Average',
    4: 'Good',
    5: 'Excellent'
}

# Create a reverse mapping from descriptive condition labels back to integer keys.
rev_condition = {v: k for k, v in condition.items()}

# for get_h3.py
output_hex_path = ""
state_shapefile = ""

# for scoring_h3.py
db_path = ""
grid_path = ""
chunk_size = 10_000_000
source_table = "condition_score"
calib_table = "condition_score_calib"
output_prefix = ""
calib_prefix = ""