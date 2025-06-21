# -*- coding: utf-8 -*-
"""
File:    scoring_h3.py
Author:  Kyusik Kim <kyusik.kim@example.com>
Date:    2025-06-20
Version: 0.1.0

Description
-----------
Load individual housing condition records from SQLite, spatially aggregate
into H3 hexagon cells, compute per‐cell counts, percentages, and weighted
condition scores, and export results as both GeoPackage and GeoJSON files
for raw and calibrated predictions.

Dependencies
------------
- Python >= 3.12
- config.py         (defines: `db_path`, `grid_path`, `chunk_size`,
                     `source_table`, `calib_table`, `output_prefix`,
                     `calib_prefix`, `rev_condition` mapping)

Usage
-----
Configure `config.py` with the following variables:
   ```python
   db_path = "path/.sqlite"
   grid_path = "path/grid.gpkg"
   chunk_size = 1_000_000
   source_table = "condition_score"
   calib_table = "condition_score_calib"
   output_prefix = "path/housing_score_hex"
   calib_prefix = "path/housing_score_hex_calib"
   rev_condition = {0: ..., 1: ..., ...}  # mapping for weighted score
"""

import sqlite3
import pandas as pd
import geopandas as gpd

import config  # defines: db_path, grid_path, chunk_size,
              # source_table, calib_table,
              # output_prefix, calib_prefix, rev_condition


def load_grid() -> gpd.GeoDataFrame:
    """Load the H3 cell grid."""
    return gpd.read_file(config.grid_path)

def fetch_and_join(table: str, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Read records from `table` in chunks, turn LAT/LON into points,
    spatially join to `grid`, and return a concatenated DataFrame.
    """
    conn = sqlite3.connect(config.db_path)
    dfs = []
    query = f"SELECT * FROM {table};"
    # query = f"SELECT PROPID, CONDITION_CAT, CONDITION_COMPLETE, CONDITION_NUM, LAT, LON, h3id, resolution FROM {table};"
    for chunk in pd.read_sql_query(query, conn, chunksize=config.chunk_size):
        pts = gpd.GeoDataFrame(chunk,
                               geometry=gpd.points_from_xy(chunk.LON, chunk.LAT),
                               crs="EPSG:4326")
        joined = gpd.sjoin(pts, grid,
                           how="inner",
                           predicate="within")
        filter_cols = ['PROPID', 'CONDITION_CAT', 'CONDITION_COMPLETE', 'CONDITION_NUM', 'h3id', 'resolution']
        dfs.append(joined[filter_cols])
    conn.close()
    return pd.concat(dfs, ignore_index=True)

# Weighted average = sum(condition_num * count) / sum(count)
def weighted_avg(x):
    return (x['condition'] * x['count']).sum() / x['count'].sum()

def export(grid: gpd.GeoDataFrame, df: pd.DataFrame, prefix: str):
    """
    Given the joined records:
      - Total count per h3id (N)
      - Missing count per h3id
      - Count per CONDITION_COMPLETE per h3id (pivoted)
      - Percentage columns
      - Weighted average score per h3id
    """
    df['CONDITION_CAT'] = df['CONDITION_CAT'].fillna("Missing")
    n_house = df.groupby('h3id').size().reset_index(name='N')
    
    # Number of “Missing” houses per cell
    miss_house = df.query("CONDITION_CAT == 'Missing'").groupby('h3id').size().reset_index(name='Missing')

    # Count of houses in each condition class per cell
    group = df.groupby(['h3id','CONDITION_COMPLETE']).size().reset_index(name='count')

    # Pivot to wide form so each CONDITION_COMPLETE becomes its own column
    class_house = group.pivot(index='h3id', columns='CONDITION_COMPLETE', values='count').reset_index()
    class_house.columns.name = None       # remove the “CONDITION_COMPLETE” label above columns
    class_house = class_house.fillna(0)

    # Merge all counts into one DataFrame
    full_house = (n_house.merge(class_house, how='left', on='h3id').merge(miss_house, how='left', on='h3id').fillna(0))
    cols = full_house.columns[2:].to_list()
    full_house[[f"{c}_pct" for c in cols]] = full_house[cols].div(full_house['N'], axis=0) * 100
    group = df.groupby(['h3id','CONDITION_COMPLETE']).size().reset_index(name='count')
    group['condition'] = group['CONDITION_COMPLETE'].map(config.rev_condition) + 1

    score_df = group.groupby('h3id').apply(weighted_avg).reset_index(name='Score')

    # Join results back onto your grid
    grid = grid.merge(score_df, on='h3id', how='left')
    out = grid.merge(full_house, on='h3id', how='left')

    cols4decimal = ['Score', 'Average_pct', 'Excellent_pct', 'Fair_pct', 'Good_pct', 'Poor_pct', 'Unsound_pct', 'Missing_pct']
    for col in cols4decimal:
        out[col] = out[col].round(3)

    out.to_file(f"{prefix}.gpkg", driver="GPKG")
    out.to_file(f"{prefix}.geojson", driver="GeoJSON")

def main():
    grid = load_grid()
    for table, prefix in [
        (config.source_table, config.output_prefix),
        (config.calib_table, config.calib_prefix)
    ]:
        df = fetch_and_join(table, grid)
        export(grid, df, prefix)
        # summary = summarize(df)
        # export(grid, summary, prefix)

if __name__ == "__main__":
    main()
