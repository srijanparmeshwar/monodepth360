import os, random, sys

input_path = sys.argv[1]
output_path = sys.argv[2]

def read_file(path):
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]

def write_file(path, data):
    if len(data) < 1:
        return

    with open(path, 'w') as file:
        file.write(data[0])
        file.writelines('\n' + datum for datum in data[1:])

def read_calibration(path):
    with open(path, "r") as calibration_file:
        return calibration_file.read()

def generate_names():
    train_config_filename = os.path.join(input_path, "train.config")
    validation_config_filename = os.path.join(input_path, "validation.config")
    test_config_filename = os.path.join(input_path, "test.config")

    if os.path.exists(train_config_filename) and os.path.exists(validation_config_filename) and os.path.exists(test_config_filename):
        train_config = [os.path.join(input_path, "top", path) for path in read_file(train_config_filename)]
        validation_config = [os.path.join(input_path, "top", path) for path in read_file(validation_config_filename)]
        test_config = [os.path.join(input_path, "top", path) for path in read_file(test_config_filename)]

        adv_names(train_config, validation_config, test_config)
    else:
        basic_names()

def adv_names(train_config, validation_config, test_config):
    train_filenames = []
    validation_filenames = []
    test_filenames = []

    for path in train_config:
        if os.path.isfile(os.path.join(path, "calibration.txt")):
            calibration = read_calibration(os.path.join(path, "calibration.txt"))
        else:
            calibration = "0 0 0"

        relative_path = os.path.relpath(path, os.path.join(input_path, "top"))
        train_filenames.extend([os.path.join(relative_path, os.path.splitext(filename)[0]) + " " + calibration for filename in os.listdir(path) if filename.endswith(".jpg") or filename.endswith(".png")])

    for path in validation_config:
        if os.path.isfile(os.path.join(path, "calibration.txt")):
            calibration = read_calibration(os.path.join(path, "calibration.txt"))
        else:
            calibration = "0 0 0"

        relative_path = os.path.relpath(path, os.path.join(input_path, "top"))
        validation_filenames.extend([os.path.join(relative_path, os.path.splitext(filename)[0]) + " " + calibration for filename in os.listdir(path) if filename.endswith(".jpg") or filename.endswith(".png")])

    for path in test_config:
        if os.path.isfile(os.path.join(path, "calibration.txt")):
            calibration = read_calibration(os.path.join(path, "calibration.txt"))
        else:
            calibration = "0 0 0"

        relative_path = os.path.relpath(path, os.path.join(input_path, "top"))
        test_filenames.extend([os.path.join(relative_path, os.path.splitext(filename)[0]) + " " + calibration for filename in os.listdir(path) if filename.endswith(".jpg") or filename.endswith(".png")])

    random.seed(0)
    random.shuffle(train_filenames)

    write_file(os.path.join(output_path, "train_filenames.txt"), train_filenames)
    write_file(os.path.join(output_path, "validation_filenames.txt"), validation_filenames)
    write_file(os.path.join(output_path, "test_filenames.txt"), test_filenames)

def basic_names():
    train_path = os.path.join(output_path, "train_filenames.txt")
    test_path = os.path.join(output_path, "test_filenames.txt")

    if len(sys.argv) > 3:
        split_ratio = float(sys.argv[3])
    else:
        split_ratio = 0.9

    random.seed(0)

    all_filenames = []
    for path, subdirectories, filenames in os.walk(os.path.join(input_path, "top")):
        if not subdirectories:
            images = [filename for filename in filenames if filename.endswith(".jpg") or filename.endswith(".png")]

            if os.path.isfile(os.path.join(path, "calibration.txt")):
                calibration = read_calibration(os.path.join(path, "calibration.txt"))
            else:
                calibration = "0 0 0"

            relative_path = os.path.relpath(path, os.path.join(input_path, "top"))
            all_filenames.extend([os.path.splitext(os.path.join(relative_path, filename))[0] + " " + calibration for filename in images if os.path.isfile(os.path.join(input_path, "bottom", relative_path, filename))])

    # Split into train and test sets.
    split_index = int(split_ratio * len(all_filenames))
    train_filenames = all_filenames[:split_index]
    random.shuffle(train_filenames)
    test_filenames = all_filenames[(split_index + 1):]

    if len(train_filenames) < 1:
        exit(0)

    # Write training filenames.
    with open(train_path, 'w') as train_file:
        train_file.write(train_filenames[0])
        train_file.writelines('\n' + filename for filename in train_filenames[1:])

    if len(test_filenames) < 1:
        exit(0)

    # Write testing filenames.
    with open(test_path, 'w') as test_file:
        test_file.write(test_filenames[0])
        test_file.writelines('\n' + filename for filename in test_filenames[1:])

if __name__ == "__main__":
    generate_names()