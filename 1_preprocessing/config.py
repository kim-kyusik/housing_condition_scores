# Configuration
import os

# Set working directory (wd)
wd = ""
os.chdir(wd)
print(f"Set working directory: {wd}")

db = "put_your_database_path"
table_name = "table_name_of_database"

ruca_file_loc = 'RUCA.csv'
bg_census_file = "blockgroup_geo.gpkg"
tract_census_file = "censustract_geo.gpkg"
all_census_tracts = "all_census_tracts.gpkg"

var_list_file = 'data/tidy/variables_list.json'

# 3_concat_data.py
ruca_table = 'RUCA2010'
ACS_bg_table = "ACS_blockgroup"
ACS_tract_table = "ACS_tract"

final_data_table_name = "table_name_finalized"
csv_filename = 'training_dataset.csv'

final_data_export_folder = 'folder_name_to_hpc'

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

# Create folder
ensure_directory('data/tidy')
ensure_directory('data/geo')
ensure_directory('data/tidy/to_hpc')
ensure_directory('report')
ensure_directory('input_data')