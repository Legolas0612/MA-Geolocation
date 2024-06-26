import csv

def convert_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        i = 0
        for line in infile:
            print(i)
            i += 1
            values = line.strip().strip('[]').split(',')
            writer.writerow([float(value) for value in values])


convert_to_csv('test_coordinates.txt', 'test_coordinates.csv')
convert_to_csv('train_coordinates.txt', 'train_coordinates.csv')

