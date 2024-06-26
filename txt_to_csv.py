import pandas as pd
import numpy as np

def txt_to_csv(filename, csv_filename):
    inputs = []
    i = 0
    print("test2")
    with open(filename, 'r') as f:
        for line in f:
            a = np.array(line.strip().split(','), dtype=np.uint8)
            if len(line) != 319488:
                a = a[:319488]
            inputs.append(a)
            print(i)
            i += 1
    inputs = np.array(inputs)
    df = pd.DataFrame(inputs)
    df.to_csv(csv_filename, index=False)

txt_to_csv('train.txt', 'train.csv')
txt_to_csv('test.txt', 'test.csv')
txt_to_csv('train_coordinates.txt', 'train_coordinates.csv')
txt_to_csv('test_coordinates.txt', 'test_coordinates.csv')
