import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def read_image(image_path, shape):
    if image_path.lower().endswith("png"):
        image = tf.image.decode_png(tf.read_file(image_path))
    else:
        image = tf.image.decode_jpeg(tf.read_file(image_path))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, shape, tf.image.ResizeMethod.AREA)
    return tf.expand_dims(image, 0)

def encode_image(image, type = "jpg"):
    quantized_image = tf.image.convert_image_dtype(image[0, :, :, :], tf.uint8)
    if type == "png":
        return tf.image.encode_png(quantized_image)
    else:
        return tf.image.encode_jpeg(quantized_image)

def write_image(image_data, filename):
    with open(filename, "w") as image_file:
        image_file.write(image_data)

# Depth image utilities taken from https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
def gray2rgb(im, cmap='plasma'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth(depth, pc=95, cmap='plasma'):
    # Convert to inverse depth.
    depth = 1.0 / (depth + 1e-6)
    depth = depth / (np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    return depth

def process_depth(depth, session):
    tf_depth = tf.expand_dims(tf.convert_to_tensor(normalize_depth(depth[0, :, :, 0])), 0)
    return session.run(encode_image(tf_depth))