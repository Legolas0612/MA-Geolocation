import pandas as pd
import folium

df = pd.read_csv('results_sorted.csv')


furthest_1000 = df.tail(1000)
closest_1000 = df.head(1000)


average_lat_furthest = furthest_1000['latitude_targets'].mean()
average_lon_furthest = furthest_1000['longitude_targets'].mean()
map_furthest = folium.Map(location=[average_lat_furthest, average_lon_furthest], zoom_start=2)

map_closest = folium.Map(location=[average_lat_furthest, average_lon_furthest], zoom_start=2)

for _, row in furthest_1000.iterrows():
    folium.Marker(
        location=[row['latitude_predictions'], row['longitude_predictions']],
        popup=f"Target: {row['image_name']}",
        icon=folium.Icon(color="red", icon="remove-sign"),
    ).add_to(map_furthest)

map_furthest_file = "map_furthest_prediction_1000.html"
map_furthest.save(map_furthest_file)

for _, row in closest_1000.iterrows():
    folium.Marker(
        location=[row['latitude_predictions'], row['longitude_predictions']],
        popup=f"Target: {row['image_name']}",
        icon=folium.Icon(color="green", icon="ok-sign"),
    ).add_to(map_closest)

map_closest_file = "map_closest_prediction_1000.html"
map_closest.save(map_closest_file)