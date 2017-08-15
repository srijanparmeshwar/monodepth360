import argparse
import numpy as np
import os
import tensorflow as tf

from image_utils import *
from spherical import *

def parse_args():
    # Construct argument parser.
    parser = argparse.ArgumentParser(description = "Utility to convert Synthia dataset into equirectangular format.")
    parser.add_argument("--input_rgb", type = str, help = "Input RGB directory.", required = True)
    parser.add_argument("--input_depth", type = str, help = "Input depth directory", required = True)
    parser.add_argument("--faces", type = str, help = "Names of face folders.", required = True)
    parser.add_argument("--output_path", type = str, help = "Output directory.", required = True)
    parser.add_argument("--frames", type = int, help = "Number of images.", required = True)
    parser.add_argument("--preview", help = "Preview depth images.", action = "store_true")

    arguments = parser.parse_args()
    return arguments

def pad_and_crop(images, width, height, pad_width, pad_height):
    start_width = (width - pad_width) / 2
    end_width = start_width + pad_width

    crop_images = images[:, :, start_width:end_width, :]

    top_pad = (pad_height - height) / 2
    bottom_pad = pad_height - (top_pad + height)

    pad_and_crop_images = tf.pad(crop_images, [[0, 0], [bottom_pad, top_pad], [0, 0], [0, 0]])
    return pad_and_crop_images

def convert():
    focal_length = 532.740352
    width = 1280
    height = 760

    pad_width = int(focal_length * 2.0)
    pad_height = int(focal_length * 2.0)

    tf_rgb_filenames = tf.placeholder(tf.string, [4])
    tf_depth_filenames = tf.placeholder(tf.string, [4])
    rgbs = [tf_read_png(tf_rgb_filenames[index]) for index in range(4)]
    depths = [tf_read_raw(tf_depth_filenames[index])[:, :, :, 0:1] for index in range(4)]
    rgbs.extend([tf.zeros([1, height, width, 3], tf.float32) for _ in range(2)])
    depths.extend([tf.zeros([1, height, width, 1], tf.uint16) for _ in range(2)])

    cubic_rgbs = [pad_and_crop(rgb, width, height, pad_width, pad_height) for rgb in rgbs]
    cubic_depths = [pad_and_crop(tf.cast(depth, tf.float32), width, height, pad_width, pad_height) for depth in depths]
    cubic_depths = [backproject_cubic_depth(cubic_depths[index], [1, pad_height, pad_width], face_map[index]) for index in range(6)]

    tf_equirectangular_rgb = encode_image(cubic_to_equirectangular(cubic_rgbs, [256, 512]), "png")
    tf_equirectangular_depth = cubic_to_equirectangular(cubic_depths, [256, 512])
    tf_preview_depth = encode_image(tf.log(1.0 + tf_equirectangular_depth), "png")
    tf_equirectangular_depth = tf.squeeze(tf_equirectangular_depth[:, :, :, 0])

    session = tf.Session()

    if not os.path.exists(os.path.join(arguments.output_path, "rgb")):
        os.makedirs(os.path.join(arguments.output_path, "rgb"))
    if not os.path.exists(os.path.join(arguments.output_path, "depth")):
        os.makedirs(os.path.join(arguments.output_path, "depth"))
    if not os.path.exists(os.path.join(arguments.output_path, "preview")):
        os.makedirs(os.path.join(arguments.output_path, "preview"))

    for index in range(arguments.frames):
        rgb_filenames = [os.path.join(arguments.input_rgb, face, "{:06}.png".format(index)) for face in arguments.faces.split(",")]
        depth_filenames = [os.path.join(arguments.input_depth, face, "{:06}.png".format(index)) for face in arguments.faces.split(",")]

        if arguments.preview:
            equirectangular_rgb, equirectangular_depth, preview_depth = session.run([tf_equirectangular_rgb, tf_equirectangular_depth, tf_preview_depth],
                                           feed_dict = {tf_rgb_filenames: rgb_filenames, tf_depth_filenames: depth_filenames})
            write_image(equirectangular_rgb, os.path.join(arguments.output_path, "rgb", "{:06}.png".format(index)))
            write_image(preview_depth, os.path.join(arguments.output_path, "preview", "{:06}.png".format(index)))
            np.save(os.path.join(arguments.output_path, "depth", "{:06}.npy".format(index)), equirectangular_depth)
        else:
            equirectangular_rgb, equirectangular_depth = session.run([tf_equirectangular_rgb, tf_equirectangular_depth],
                                           feed_dict = {tf_rgb_filenames: rgb_filenames, tf_depth_filenames: depth_filenames})
            write_image(equirectangular_rgb, os.path.join(arguments.output_path, "rgb", "{:06}.png".format(index)))
            np.save(os.path.join(arguments.output_path, "depth", "{:06}.npy".format(index)), equirectangular_depth)

if __name__ == "__main__":
    arguments = parse_args()
    convert()