import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from shapely.errors import GEOSException

f = lambda x: (1 - ((x ** 0.5) / (20000000 ** 0.5))) ** 2

shapefile_path = 'c1976_2000_0/c1976_2000.shp'
gdf = gpd.read_file(shapefile_path)

print(f"Original CRS: {gdf.crs}")

coordinates_df = pd.read_csv('Datasets/geolocation/images.csv', usecols=['name', 'latitude', 'longitude'])

points = [Point(xy) for xy in zip(coordinates_df['longitude'], coordinates_df['latitude'])]

if gdf.crs.is_geographic:
    gdf = gdf.to_crs(epsg=32633)

projected_points = gpd.GeoSeries(points, crs="EPSG:4326").to_crs(gdf.crs)

def calculate_distance(geom, point):
    if geom.is_empty or point.is_empty:
        return float('inf')
    return geom.distance(point)

distances = []
valid_rows = []
error = []
for idx, point in enumerate(projected_points):
    try:
        distance_row = gdf.geometry.apply(lambda geom: f(calculate_distance(geom, point))).tolist()
        distances.append(distance_row)
        valid_rows.append(idx)
    except GEOSException as e:
        print(f"Skipping point at index {idx} due to error: {e}")
        error.append(idx)

with open('error_points.txt', 'w') as f:
    for idx in error:
        f.write(f"{coordinates_df.iloc[idx]['name']}\n")

valid_coordinates_df = coordinates_df.iloc[valid_rows].reset_index(drop=True)

distance_df = pd.DataFrame(distances, columns=[f'distance_to_polygon_{i+1}' for i in range(len(gdf))])

output_df = pd.concat([valid_coordinates_df, distance_df], axis=1)

output_df.to_csv('distance_to_point.csv', index=False)

print(f"Transformed CRS: {gdf.crs}")
print(output_df.head())