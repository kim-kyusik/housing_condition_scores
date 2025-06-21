# Configuration
import os

# Set working directory (wd)
wd = ""
os.chdir(wd)
print(f"Set working directory: {wd}")

db = "D:/sqlite_db/Real_Estate_Property_Data_Warren_12_23_2024.sqlite"
table_name = "residential_whole_us_data"

ruca_file_loc = 'data/raw/RUCA_2010.csv'
bg_census_file = "data/geo/census_info_bg.gpkg"
tract_census_file = "data/geo/census_info_tract.gpkg"

var_list_file = 'data/tidy/variables_list.json'

# 3_concat_data.py
ruca_table = 'RUCA2010'
ACS_bg_table = "ACS_blockgroup"
ACS_tract_table = "ACS_tract"

final_data_table_name = "data_final_v25.02.02"
csv_filename = 'data/tidy/training_data_set.csv'

final_data_export_folder = 'data/tidy/to_hpc'

def ensure_directory(directory):
    """
    Check if a directory exists, and if it does not, create it.
    
    Parameters:
        directory (str): The path to the directory.
    
    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Example usage:
ensure_directory('data/tidy')
ensure_directory('data/geo')
ensure_directory('data/tidy/to_hpc')