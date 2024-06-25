def delete_specific_line(original_file_path, line_number=11059):
    temp_file_path = original_file_path + '.tmp'
    with open(original_file_path, 'r') as read_file, open(temp_file_path, 'w') as write_file:
        for current_line, line in enumerate(read_file, start=1):
            if current_line != line_number:
                write_file.write(line)
            print(current_line)
    import os
    os.remove(original_file_path)
    os.rename(temp_file_path, original_file_path)

file_path = 'rgb_values4.txt'
delete_specific_line(file_path)