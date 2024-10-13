import pandas as pd
from shapely.geometry import Point, shape
from fastkml import kml

def load_kml(file_path):
    with open(file_path, 'rt', encoding='utf-8') as f:
        doc = f.read()
    
    k = kml.KML()
    k.from_string(doc)
    polygons = []
    
    for feature in list(k.features()):
        for placemark in list(feature.features()):
            polygon = placemark.geometry
            zone_name = placemark.name
            polygons.append((shape(polygon), zone_name))
    
    return polygons

# Function to find the climate zone for a given point
def find_climate_zone(point, polygons):
    for polygon, zone_name in polygons:
        if polygon.contains(point):
            return zone_name
    return "Unknown"

# Load the dataset of coordinates
data = pd.read_csv('train_coordinates.csv') 

# Load the KML file
climate_zones = load_kml('Klimazonen/Global_1986-2010_KG_5m/KG_1986-2010.kml')

# Match coordinates to climate zones
data['Climate Zone'] = data.apply(lambda row: find_climate_zone(Point(row['longitude'], row['latitude']), climate_zones), axis=1)

# Save the results to a new CSV
data.to_csv('coordinates_with_climate_zones.csv', index=False)

print("Matching complete. Results saved to 'coordinates_with_climate_zones.csv'.")
