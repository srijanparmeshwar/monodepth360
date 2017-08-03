import os, random, sys

input_path = sys.argv[1]
output_path = sys.argv[2]

train_path = os.path.join(output_path, "train_filenames.txt")
test_path = os.path.join(output_path, "test_filenames.txt")

if len(sys.argv) > 3:
	split_ratio = float(sys.argv[3])
else:
	split_ratio = 0.9

def read_calibration(path):
    with open(path, "r") as calibration_file:
        return calibration_file.read()
    
all_filenames = []
for path, subdirectories, filenames in os.walk(os.path.join(input_path, "top")):
    if not subdirectories:
        images = [filename for filename in filenames if filename.endswith(".jpg") or filename.endswith(".png")]
        
        if os.path.isfile(os.path.join(path, "calibration.txt")):
            calibration = read_calibration(os.path.join(path, "calibration.txt"))
        else:
            calibration = "0 0 0"
        
        relative_path = os.path.relpath(path, os.path.join(input_path, "top"))
        all_filenames.extend([os.path.splitext(os.path.join(relative_path, filename))[0] + " " + calibration for filename in images if os.path.isfile(os.path.join("bottom", path, filename))])

if len(all_filenames) < 1:
	exit(0)

# Split into train and test sets.
split_index = int(split_ratio * len(all_filenames))
train_filenames = all_filenames[:split_index]
test_filenames = all_filenames[(split_index + 1):]

# Write training filenames.
with open(train_path, 'w') as train_file:
	train_file.write(train_filenames[0])
	train_file.writelines('\n' + filename for filename in train_filenames[1:])

# Write testing filenames.
with open(test_path, 'w') as test_file:
	test_file.write(test_filenames[0])
	test_file.writelines('\n' + filename for filename in test_filenames[1:])

