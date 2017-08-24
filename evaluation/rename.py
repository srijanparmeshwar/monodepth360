import argparse
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description = "Rename files in order.")
    parser.add_argument("--input_path", type = str, help = "Path to input data.", required = True)
    parser.add_argument("--output_path", type = str, help = "Path to output data.", required = True)
    parser.add_argument("--input_file", type = str, help = "Input filename file.", required = True)
    parser.add_argument("--input_ext", type = str, help = "Input extension.", default = ".exr")
    parser.add_argument("--output_format", type = str, help = "Format of output filenames.", required = True)

    arguments = parser.parse_args()

    return arguments

def read_file(path):
    with open(path, "r") as file:
        return [line.strip().split(" ")[0] + arguments.input_ext for line in file.readlines()]

def rename():
    relative_filenames = read_file(arguments.input_file)

    index = 0
    for relative_filename in relative_filenames:
        if "domain_rgb" in relative_filename:
            relative_filename = relative_filename.replace("domain_rgb", "domain_depth")
        input_filename = os.path.join(arguments.input_path, relative_filename)
        output_filename = os.path.join(arguments.output_path, arguments.output_format.format(index))
        shutil.copy(input_filename, output_filename)
        index += 1

if __name__ == "__main__":
    arguments = parse_args()
    rename()
