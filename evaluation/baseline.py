from __future__ import print_function

import argparse
import numpy as np
import os

from reader import read_file

def parse_args():
    parser = argparse.ArgumentParser(description = "Calculate median of medians from training set.")
    parser.add_argument("--input_path", type = str, help = "Path to training data.", required = True)
    parser.add_argument("--output_path", type = str, help = "Path to output directory.", required = True)
    parser.add_argument("--ext", type = str, help = "Depth map extension.", default = ".exr")

    arguments = parser.parse_args()

    return arguments

# Calculate median of medians from depth images in training set.
def calculate(arguments):
    filenames = [os.path.join(arguments.input_path, filename) for filename in os.listdir(arguments.input_path) if filename.endswith(arguments.ext)]
    medians = []
    for filename in filenames:
        depth_map = read_file(filename)
        median = np.nanmedian(depth_map)
        medians.append(median)

    medians = np.array(medians)

    # Check and create output directory.
    if not os.path.exists(arguments.output_path):
        os.makedirs(arguments.output_path)

    with open(os.path.join(arguments.output_path, "median.txt"), "w") as file:
        file.write("{:.6f}".format(np.median(medians)))
        print("{:.6f}".format(np.median(medians)))

if __name__ == "__main__":
    arguments = parse_args()
    calculate(arguments)