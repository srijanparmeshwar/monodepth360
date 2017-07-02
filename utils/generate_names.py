import os, random, sys

data_path = sys.argv[1]
filename_file_path = sys.argv[2]

if len(sys.argv) > 3:
	random.seed(sys.argv[3])
else:
	random.seed(0)

filenames = os.listdir(os.path.join(data_path, '2d'))
# [1.jpg, 2.jpg, ...] -> [1, 2, ...]
stripped_filenames = [os.path.splitext(filename)[0] for filename in filenames]
random.shuffle(stripped_filenames)

if len(stripped_filenames) < 1:
	exit(0)

with open(filename_file_path, 'w') as filename_file:
	filename_file.write(stripped_filenames[0])
	filename_file.writelines('\n' + filename for filename in stripped_filenames[1:])
