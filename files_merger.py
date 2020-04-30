import sys
import os

data_dir = sys.argv[1]
file_names = sys.argv[2]
file_names = file_names.split(',')

all_lines = []
for file_name in file_names:
    all_lines.append(open(os.path.join(data_dir, file_name)).readlines())

for line_idx in range(len(all_lines[0])):
    for file_idx in range(len(all_lines)):
        print(all_lines[file_idx][line_idx].strip())
    print('')