import pandas as pd

def remove_values_below_threshold(df, threshold):
    rows_to_ignore = ["name", "latitude", "longitude"]
    for column in df.columns:
        if column not in rows_to_ignore:
            df[column] = df[column].apply(lambda x: x if x > threshold else 0)
    return df

df = pd.read_csv('distance_to_point_test.csv')
df = remove_values_below_threshold(df, 0.1)
df.to_csv('distance_to_point_test.csv', index=False)

df = pd.read_csv('distance_to_point_train.csv')
df = remove_values_below_threshold(df, 0.1)
df.to_csv('distance_to_point_train.csv', index=False)