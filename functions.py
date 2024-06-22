def get_rgb_values(id):
    import PIL as pil
    from PIL import Image
    try:
        img = Image.open(f"C:\\Users\\Fabian Schmid\\.vscode\\Programme\\Maturarbeit\\Datasets\\geolocation\\images\\{id}.jpeg")
    except:
        with open('error2.txt', 'a') as f:
            f.write(id)
        return []
    #img = Image.open(f"C:\\Users\\Fabian Schmid\\.vscode\\Programme\\Maturarbeit\\Datasets\\geolocation\\images\\{id}.jpeg")

    rgb_values = list(img.getdata())
    output = []
    for rgb in rgb_values:
        pixel = []
        for color in rgb:
            pixel.append(color)
        output.append(pixel)
    return output
        

def get_imageId_latitude_longitude_values():
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load spreadsheet
    xl = pd.ExcelFile(r"C:\Users\Fabian Schmid\.vscode\Programme\Maturarbeit\Datasets\geolocation\images.xlsx")

    # Load a sheet into a DataFrame by its name
    df = xl.parse('images')

    # Split the strings at the commas
    df = df.iloc[:, 0].str.split(',', expand=True)

    # Select features and target
    image_id = df.iloc[:, 0].values
    latitude = df.iloc[:, 1].values
    longitude = df.iloc[:, 2].values

    # Convert latitude and longitude to numeric values
    latitude = latitude.astype(float)
    longitude = longitude.astype(float)

    return image_id, latitude, longitude