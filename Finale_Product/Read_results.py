import pandas as pd
from geopy.distance import geodesic

df = pd.read_csv('results.csv')

def calculate_distance(row):
    target_coords = (row['latitude_targets'], row['longitude_targets'])
    pred_coords = (row['latitude_predictions'], row['longitude_predictions'])
    return geodesic(target_coords, pred_coords).meters

df['distance'] = df.apply(calculate_distance, axis=1)

df_sorted = df.sort_values(by='distance')

mean_distance = df['distance'].mean()

top_50_threshold = df_sorted['distance'].quantile(0.50)
top_25_threshold = df_sorted['distance'].quantile(0.25)

top_10_threshold = df_sorted['distance'].quantile(0.10)

print(f"Average Distance: {mean_distance:.2f} meters")
print(f"Distance to be in the Top 50%: {top_50_threshold:.2f} meters")
print(f"Distance to be in the Top 25%: {top_25_threshold:.2f} meters")
print(f"Distance to be in the Top 10%: {top_10_threshold:.2f} meters")

# df_sorted.to_csv('results_sorted.csv', index=False)
