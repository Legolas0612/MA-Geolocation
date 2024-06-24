from functions import get_rgb_values, get_imageId_latitude_longitude_values

image_id, latitude, longitude = get_imageId_latitude_longitude_values()
with open('coordinates.txt', 'a') as f:
    for i in range(len(latitude)):
        f.write(str(latitude[i]) + ',' + str(longitude[i]) + '\n')
        continue

# Get the RGB values for each image
rgb_values = []
i = 0
for id in image_id:
    rgb_values = get_rgb_values(str(id))
    #print(rgb_values)
    with open('rgb_values4.txt', 'a') as f:
        # Write each value on a new line
        rgb_values = str(rgb_values).replace(']', '').replace('[', '').replace(" ", "")
        f.write(str(rgb_values))
        f.write('\n')
    i += 1
    print(i)
    with open('counter3.txt', 'a') as f:
        f.write(str(i))
        f.write('\n')
        continue