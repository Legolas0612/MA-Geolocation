def read_specific_line(file_path, line_number=11059):
    with open(file_path, 'r') as file:
        for current_line, line in enumerate(file, start=1):
            if current_line == line_number:
                return line.strip()  # Removes any leading/trailing whitespace and newline characters
            print(current_line)
    

# Example usage
#file_path = 'rgb_values4.txt'
#specific_line = read_specific_line(file_path)
#print(specific_line)
tmp = 0
with open('rgb_values4.txt', 'r') as f:
    for i, line in enumerate(f):
        a = line.strip().split(',')
        print(i, len(a))
        tmp = len(a)            
