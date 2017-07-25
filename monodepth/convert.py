import argparse
import os

from image_utils import *
from spherical import *

# Construct argument parser.
parser = argparse.ArgumentParser(description='Image converter for spherical formats.')
parser.add_argument("--mode", type = str, help = "cubic2equirectangular or equirectangular2cubic.", default = "equirectangular2cubic")
parser.add_argument("--input_path", type = str, help = "Path to directory containing input images.", default = "./")
parser.add_argument("--output_path", type = str, help = "Path to output directory.", default = "/output/")
parser.add_argument("--input_format", type = str, help = "Input format - either jpg or png.", default = "jpg")
parser.add_argument("--output_format", type = str, help = "Output format - either jpg or png.", default = "png")
parser.add_argument("--input_height", type = int, help = "Input height.", default = 2048)
parser.add_argument("--input_width", type = int, help = "Input width.", default = 4096)
parser.add_argument("--output_height", type = int, help = "Output height.", default = 1024)
parser.add_argument("--output_width", type = int, help = "Output width.", default = 1024)
parser.add_argument("--faces", type = str, help = "Which cube faces to output.", default = "0")

# Parse arguments.
arguments = parser.parse_args()

# Find and modify filenames in input directory.
def get_filenames():
    filenames = [filename for filename in os.listdir(arguments.input_path) if filename.lower().endswith(arguments.input_format)]
    input_files = [os.path.join(arguments.input_path, filename) for filename in filenames]
    output_files = [os.path.join(arguments.output_path, filename[:-4] + "." + arguments.output_format) for filename in filenames]
    return input_files, output_files

def e2c(session):
    input_files, output_files = get_filenames()
    face_indices = [int(index) for index in arguments.faces.split(",")]
    num_faces = len(face_indices)
    for image_index in range(len(input_files)):
        input_filename = input_files[image_index]
        output_filename = output_files[image_index]
        # Load equirectangular image.
        equirectangular_image = read_image(input_filename, [arguments.input_height, arguments.input_width])
        faces = equirectangular_to_cubic(equirectangular_image, [arguments.output_height, arguments.output_width])
        image_data = session.run([encode_image(faces[index]) for index in face_indices])
        # Iterate over and write faces.
        for index in range(num_faces):
            write_image(image_data[index], output_filename[:-4] + "_" + face_map[index] + output_filename[-4:])

def c2e(session):
    input_files, output_files = get_filenames()
    # Not implemented yet.

if __name__ == "__main__":
    session = tf.Session()
    if arguments.mode == "cubic2equirectangular":
        c2e(session)
    else:
        e2c(session)