# -*- coding: utf-8 -*-
"""
File:    get_h3.py
Author:  Kyusik Kim <kyusik.kim@example.com>
Date:    2025-06-20
Version: 0.1.0

Description
-----------
Generate an H3 hexagon grid covering all U.S. states using the H3 library. Reads
state boundaries from a shapefile, converts them to H3 cell IDs at a specified
resolution, builds polygons for each cell, and exports the result as a GeoPackage.

Dependencies
------------
- Python >= 3.12
- config.py (defines `state_shapefile`: path to state boundaries shapefile, and
  `output_hex_path`: path for output GeoPackage)

Usage
-----
In `config.py`, set:
   ```python
   state_shapefile = "path/to/tl_2023_us_state.shp"
   output_hex_path = "path/to/source_geo_hex.gpkg"
"""

import h3
import geopandas as gpd
from shapely.geometry import Polygon
from us import states
import config  # Should define state_shapefile and output_hex_path


from us import states
all_state = []
for state in states.STATES:
    all_state.append(state.fips)
all_state.append(states.DC.fips)

census_geo = gpd.read_file(config.state_shapefile)
source_geo = census_geo.query("STATEFP in @all_state") # 51 states

def swap_latlon(polygon):
    # This function swaps the coordinates in the polygon's exterior ring.
    # It assumes the incoming polygon has coordinates in (lat, lon) order and produces
    # a new Polygon with coordinates in (lon, lat) order (which is standard for Shapely).
    swap = [(lon, lat) for lat, lon in polygon.exterior.coords]
    return Polygon(swap)

def create_h3_hex(gdf, res):
    # Reset the index of the GeoDataFrame in case it has a non-default index.
    gdf = gdf.reset_index()

    # This assumes that the GeoDataFrame (gdf) contains a single polygon geometry.
    # NOTE: The H3 resolution is hard-coded as 7 in this call even though a 'res' parameter is provided.
    # It might be preferable to use the 'res' parameter here (e.g., res=res).

    if len(gdf) > 1:
        h3_cells = []
        for i in gdf.geometry:
            h3_cell = h3.geo_to_cells(geo=i, res=res)
            h3_cells = h3_cells + h3_cell
    else:
        h3_cells = h3.geo_to_cells(geo=gdf.geometry[0], res=res)  # Consider: use res=res

    # For each H3 cell, convert the cell boundary into a Shapely Polygon.
    # Note: h3.cell_to_boundary(x) returns the boundary coordinates.
    geom = [Polygon(h3.cell_to_boundary(x)) for x in h3_cells]

    # Apply swap_latlon to each hexagon geometry to swap coordinate order,
    # ensuring that coordinates are in (lon, lat) for proper plotting/usage in GeoPandas.
    h3_geom = [swap_latlon(x) for x in geom]

    # Create an attribute dictionary for the GeoDataFrame.
    # It includes the H3 cell IDs and the resolution.
    data = {'h3id': h3_cells, 'resolution': res}
    
    # Return a new GeoDataFrame with the hexagon geometries and associated attributes.
    return gpd.GeoDataFrame(data, geometry=h3_geom, crs="EPSG:4326")

res = create_h3_hex(source_geo, 6)
res.to_file(config.output_hex_path, driver="GPKG")