import argparse
import os
import tensorflow as tf

from image_utils import *
from spherical import *

def parse_args():
    # Construct argument parser.
    parser = argparse.ArgumentParser(description = "Image converter for spherical formats.")
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
    parser.add_argument("--batch_size", type = int, help = "Batch size for processing.", default = 8)

    # Parse arguments.
    return parser.parse_args()

# Find and modify filenames in input directory.
def get_filenames(arguments):
    filenames = [filename for filename in os.listdir(arguments.input_path) if filename.lower().endswith(arguments.input_format)]
    input_files = [os.path.join(arguments.input_path, filename) for filename in filenames]
    output_files = [os.path.join(arguments.output_path, filename[:-4] + "." + arguments.output_format) for filename in filenames]
    return input_files, output_files

def e2c(arguments):
    input_files, output_files = get_filenames(arguments)
    face_indices = [int(index) for index in arguments.faces.split(",")]
    num_faces = len(face_indices)
    image_index = 0
    while image_index < len(input_files):
        input_filenames = input_files[image_index:min(image_index + arguments.batch_size, len(input_files))]
        output_filenames = output_files[image_index:min(image_index + arguments.batch_size, len(input_files))]
        with tf.Graph().as_default(), tf.Session() as session:
            equirectangular_images = tf.concat([read_image(input_filename, [arguments.input_height, arguments.input_width]) for input_filename in input_filenames], 0)
            faces = equirectangular_to_cubic(equirectangular_images, [arguments.output_height, arguments.output_width])
            image_data = session.run([encode_images(faces[index], min(arguments.batch_size, len(input_files) - image_index)) for index in face_indices])
            for index in range(num_faces):
                for output_index in range(len(output_filenames)):
                    write_image(image_data[index][output_index], output_filenames[output_index][:-4] + "_" + face_map[face_indices[index]] + output_filenames[output_index][-4:])
        
        image_index += arguments.batch_size

def c2e(arguments):
    input_files, output_files = get_filenames(arguments)
    # Not implemented yet.

if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "cubic2equirectangular":
        c2e(arguments)
    else:
        e2c(arguments)