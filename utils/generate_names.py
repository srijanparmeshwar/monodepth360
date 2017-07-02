import os, random, sys

data_path = sys.argv[1]
output_path = sys.argv[2]
train_path = output_path + "train_filenames.txt"
test_path = output_path + "test_filenames.txt"

if len(sys.argv) > 3:
	random.seed(sys.argv[3])
else:
	random.seed(0)

if len(sys.argv) > 4:
	split_ratio = float(sys.argv[4])
else:
	split_ratio = 0.9

# Find all 2D image filenames in input path.
all_filenames = os.listdir(os.path.join(data_path, '2d'))

if len(all_filenames) < 1:
	exit(0)

# Remove file extenstions and shuffle order.
stripped_filenames = [os.path.splitext(filename)[0] for filename in all_filenames]
random.shuffle(stripped_filenames)

# Split into train and test sets.
split_index = int(split_ratio * len(stripped_filenames))
train_filenames = stripped_filenames[:split_index]
test_filenames = stripped_filenames[(split_index + 1):]

# Write training filenames.
with open(train_path, 'w') as train_file:
	train_file.write(train_filenames[0])
	train_file.writelines('\n' + filename for filename in train_filenames[1:])

# Write testing filenames.
with open(test_path, 'w') as test_file:
	test_file.write(test_filenames[0])
	test_file.writelines('\n' + filename for filename in test_filenames[1:])

