from functions import get_rgb_values
import pandas as pd

image_id = []

df = pd.read_csv('distance_to_point.csv')
image_id = df['name'].tolist()

rgb_values = []
i = 0
with open("rgb_values_climatezone.csv", "a") as f:
    for id in image_id:
        rgb_values = get_rgb_values(str(id))
        rgb_values = str(rgb_values).replace(']', '').replace('[', '').replace(" ", "")
        f.write(str(rgb_values))
        f.write('\n')
        i += 1
        if i % 1000 == 0:
            print(i)
