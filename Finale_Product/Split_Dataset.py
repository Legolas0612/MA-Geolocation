import pandas as pd

def split_dataset(coordinates_path, ratio=0.1):
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


csv_path = 'distance_to_point_No_inf.csv'
split_dataset(csv_path)