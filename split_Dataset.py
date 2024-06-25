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

coordinates = 'coordinates.txt'
rgb_values = 'rgb_values4.txt'

split_dataset(coordinates, rgb_values)
