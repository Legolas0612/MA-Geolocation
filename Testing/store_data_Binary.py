import struct


with open('train.txt', 'r') as file, open('train.bin', 'wb') as bin_file:
    i = 0
    for line in file:
        print(i)
        i += 1
        line = line.strip().strip('[]')
        values = line.split(',')
        for value in values:
            grayscale_value = int(value.strip().strip("'"))
            bin_file.write(struct.pack('B', grayscale_value))


