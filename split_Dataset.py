import pandas as pd

def split_dataset(coordinates_path, rgb_values_path):
    with open(rgb_values_path, 'r') as f, \
         open(coordinates_path, 'r') as coordinates_file, \
         open('train.txt', 'w') as train, \
         open('test.txt', 'w') as test, \
         open("train_coordinates.txt", "w") as train_coordinates, \
         open("test_coordinates.txt", "w") as test_coordinates:
        
        for current_line, line in enumerate(f, start=1):
            if current_line <= 100000:
                train.write(line)
            else:
                test.write(line)
            print(current_line, "rgb_values")
        
        for current_line, line in enumerate(coordinates_file, start=1):
            if current_line <= 100000:
                train_coordinates.write(line)
            else:
                test_coordinates.write(line)
            print(current_line, "coordinates")

def split_new_dataset(coordinates_path, ratio=0.1):
    df = pd.read_csv(coordinates_path)
    num_samples = len(df)
    test_size = min(ratio, max(ratio, (num_samples - 1) / num_samples))
    train_size = 1 - test_size
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv('distance_to_point_train.csv', index=False)
    test_df.to_csv('distance_to_point_test.csv', index=False)

def train_test_split(df, test_size=0.1, random_state=None):
    if random_state:
        df = df.sample(frac=1, random_state=random_state)
    num_samples = len(df)
    test_size = min(test_size, max(test_size, (num_samples - 1) / num_samples))
    test_size = int(test_size * num_samples)
    test_df = df[:test_size]
    train_df = df[test_size:]
    return train_df, test_df

coordinates = 'coordinates.txt'
rgb_values = 'rgb_values4.txt'

#split_dataset(coordinates, rgb_values)

csv_path = 'distance_to_point_No_inf.csv'
split_new_dataset(csv_path)