import csv
import math

LAT_DIVISIONS = 10
LON_DIVISIONS = 10

MAX_DISTANCE = math.sqrt((90)**2 + (180)**2)

def assign_part(lat, lon):
    lat_part = min(int((lat + 90) / (180 / LAT_DIVISIONS)), LAT_DIVISIONS - 1)
    lon_part = min(int((lon + 180) / (360 / LON_DIVISIONS)), LON_DIVISIONS - 1)
    return lat_part, lon_part

def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def transformation_function(distance):
    return (1 - (distance**0.5 / MAX_DISTANCE**0.5)) ** 2

def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [f'part_{i}_{j}' for i in range(LAT_DIVISIONS) for j in range(LON_DIVISIONS)]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:
            lat = float(row['latitude'])
            lon = float(row['longitude'])

            lat_part, lon_part = assign_part(lat, lon)

            part_values = {}
            for i in range(LAT_DIVISIONS):
                for j in range(LON_DIVISIONS):
                    distance = calculate_distance(lat_part, lon_part, i, j)
                    value = transformation_function(distance)
                    value = round(value, 5)
                    if value < 0.8:
                        value = 0
                    part_values[f'part_{i}_{j}'] = value

            part_values[f'part_{lat_part}_{lon_part}'] = 1

            new_row = {**row, **part_values}
            writer.writerow(new_row)

input_csv = 'Datasets/geolocation/images.csv'
output_csv = 'parts.csv'
process_csv(input_csv, output_csv)
