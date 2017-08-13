from __future__ import print_function

import os, shutil, sys

input_path = sys.argv[1]
output_path = sys.argv[2]
resize_settings = sys.argv[3]

for path, subdirectories, filenames in os.walk(input_path):
    if not subdirectories:
        paths = [filename for filename in filenames if filename.endswith(".jpg")]
        relative_path = os.path.relpath(path, input_path)
        if not os.path.exists(os.path.join(output_path, relative_path)) and len(paths) > 0:
            os.makedirs(os.path.join(output_path, relative_path))
        if os.path.exists(os.path.join(input_path, relative_path, "calibration.txt")) and not os.path.exists(os.path.join(output_path, relative_path, "calibration.txt")):
            shutil.copy(os.path.join(input_path, relative_path, "calibration.txt"), os.path.join(output_path, relative_path, "calibration.txt"))

        new_paths = [os.path.join(output_path, relative_path, filename) for filename in paths]
        for index in range(len(paths)):
            new_path = new_paths[index]
            command = "convert {} -resize {} {}".format(os.path.join(path, paths[index]), resize_settings, new_path)
            if not os.path.exists(new_path):
                print(command)
                os.system(command)
