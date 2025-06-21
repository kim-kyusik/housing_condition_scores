# -*- coding: utf-8 -*-
"""
File:    add_external_vars.py
Author:  Kyusik Kim <kyusik.q.kim@gmail.com>
Date:    2025-06-20
Version: 0.1.0

Description
------------
Append external geographic and socioeconomic variables by performing spatial joins. 
Specifically, it:
    1. Joins property points with census tract geometries to assign RUCA codes
        and writes to table `RUCA2010`.
    2. Joins with ACS Block Group polygons to append block‐group level attributes
        and writes to table `ACS_blockgroup`.
    3. Joins with ACS Tract polygons to append tract‐level attributes (e.g.,
        poverty_pct) and writes to table `ACS_tract`.

Dependencies
------------
- Python >= 3.12
- config.py    (must define `db`, `table_name`, `ruca_file_loc`, `bg_census_file`, `tract_census_file`)

Usage
-----
1. Edit `config.py` to point at your database and data files, for example:
   ```python
   db                = "path/to/database.db"
   table_name        = "properties"
   ruca_file_loc     = "data/ruca.csv"
   bg_census_file    = "data/acs_blockgroups.gpkg"
   tract_census_file = "data/acs_tracts.gpkg"
"""

# Import packages
import sqlite3
import pandas as pd
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
import config

# Connect to the database and create a cursor.
conn = sqlite3.connect(config.db)
cursor = conn.cursor()
table_name = config.table_name

# --------- RUCA: Spatial Join with Census Tract Data ---------
# Read geographic census tract data (GeoPackage)
geo_data = gpd.read_file(r"data/geo/all_census_tracts.gpkg")

# Load RUCA CSV file, ensuring string type for GEOID and RUCA fields,
# and select only the GEOID10 and RUCA2010 columns.
ruca = pd.read_csv(config.ruca_file_loc, dtype={'GEOID': str, 'RUCA': str})
ruca['GEOID10'] = ruca['GEOID']
ruca = ruca[['GEOID10', 'RUCA2010']]

# Define source and destination table names.
source_table = table_name
new_table = 'RUCA2010'

# Query: select property id and coordinates for spatial join.
cols = "PROPID, LAT, LON"
query = f"SELECT {cols} FROM {source_table};"

# Process data in chunks for efficient memory use.
chunk_size = 100000
chunk_running = 0

for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
    # Create geometry from longitude and latitude.
    chunk['geometry'] = [Point(lon, lat) for lon, lat in zip(chunk['LON'], chunk['LAT'])]
    chunk = gpd.GeoDataFrame(chunk, geometry='geometry', crs="EPSG:4326")
    
    # Reproject chunk to match census tract CRS if necessary.
    if chunk.crs != geo_data.crs:
        chunk = chunk.to_crs(geo_data.crs)
    
    # Perform spatial join between properties and census tract data.
    joined = gpd.sjoin(chunk, geo_data, how='left', predicate='intersects')
    
    # Keep relevant columns.
    cols_keep = ['PROPID', 'STATEFP10', 'COUNTYFP10', 'TRACTCE10', 'GEOID10']
    joined = joined[cols_keep]
    
    # Merge with RUCA information based on GEOID10.
    chunk_result = pd.merge(joined, ruca, how='left', on='GEOID10')
    
    chunk_running += chunk_size
    print(f"Appending: {chunk_running}")
    
    # Append the results to the RUCA2010 table in the database.
    chunk_result.to_sql(new_table, conn, if_exists='append', index=False)

# --------- ACS Information: Block Group Level ---------

# Load ACS Block Group geographic data.
geo_data = gpd.read_file(config.bg_census_file)
# Rename the GEOID column to GEOID12 and drop unwanted columns.
geo_data = geo_data.rename(columns={'GEOID': 'GEOID12'}).drop(columns=['state', 'county', 'tract', 'block group'])
# Get non-geometry columns to store.
geo_cols = list(geo_data.columns)
geo_cols.remove('geometry')

new_table = 'ACS_blockgroup'
query = f"SELECT PROPID, LAT, LON FROM {source_table};"
chunk_size = 1000000
chunk_running = 0

for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
    # Create a geometry column.
    chunk['geometry'] = [Point(lon, lat) for lon, lat in zip(chunk['LON'], chunk['LAT'])]
    chunk = gpd.GeoDataFrame(chunk, geometry='geometry', crs="EPSG:4326")
    
    # Reproject chunk to match geo_data CRS.
    if chunk.crs != geo_data.crs:
        chunk = chunk.to_crs(geo_data.crs)
    
    # Spatial join with block group data.
    joined = gpd.sjoin(chunk, geo_data, how='left', predicate='intersects')
    
    # Select required columns.
    joined = joined[['PROPID'] + geo_cols]
    
    chunk_running += chunk_size
    print(f"Appending: {chunk_running}")
    
    # Append to the ACS_blockgroup table.
    joined.to_sql(new_table, conn, if_exists='append', index=False)

# --------- ACS Information: Tract Level ---------

# Load ACS Tract level data and rename GEOID to GEOID11.
geo_data = gpd.read_file(config.tract_census_file)
geo_data = geo_data.rename(columns={'GEOID': 'GEOID11'})
geo_cols = list(geo_data.columns)
geo_cols.remove('geometry')

new_table = 'ACS_tract'
query = f"SELECT PROPID, LAT, LON FROM {source_table};"
chunk_size = 1000000
chunk_running = 0

for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
    # Create geometry using LON and LAT.
    chunk['geometry'] = [Point(lon, lat) for lon, lat in zip(chunk['LON'], chunk['LAT'])]
    chunk = gpd.GeoDataFrame(chunk, geometry='geometry', crs="EPSG:4326")
    
    # Reproject if necessary.
    if chunk.crs != geo_data.crs:
        chunk = chunk.to_crs(geo_data.crs)
    
    # Spatial join between properties and tract data.
    joined = gpd.sjoin(chunk, geo_data, how='left', predicate='intersects')
    
    # Keep required columns.
    joined = joined[['PROPID', 'GEOID11', 'poverty_pct']]
    
    chunk_running += chunk_size
    print(f"Appending: {chunk_running}")
    
    # Append results to the ACS_tract table.
    joined.to_sql(new_table, conn, if_exists='append', index=False)