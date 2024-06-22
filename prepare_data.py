from functions import get_rgb_values, get_imageId_latitude_longitude_values

image_id, latitude, longitude = get_imageId_latitude_longitude_values()

# Get the RGB values for each image
rgb_values = []
i = 0
for id in image_id:
    rgb_values = get_rgb_values(str(id))
    with open('rgb_values3.txt', 'a') as f:
    # Write each value on a new line
        for value in rgb_values:
            value = str(value).replace('][', ';').replace(']', '').replace('[', '').replace(" ", "")
            f.write(str(value))
        f.write('\n')
    i += 1
    print(i)
    with open('counter3.txt', 'a') as f:
        f.write(str(i))
        f.write('\n')