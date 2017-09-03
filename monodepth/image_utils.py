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

def tf_read_png(image_path):
    image = tf.image.decode_png(tf.read_file(image_path))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.expand_dims(image, 0)

def tf_read_raw(image_path):
    image = tf.image.decode_png(tf.read_file(image_path), dtype = tf.uint16)
    return tf.expand_dims(image, 0)

def encode_image(image, type = "jpg", index = 0):
    quantized_image = tf.image.convert_image_dtype(image[index, :, :, :], tf.uint8)
    if type == "png":
        return tf.image.encode_png(quantized_image)
    else:
        return tf.image.encode_jpeg(quantized_image)
        
def encode_images(images, batch_size, type = "jpg"):
    return [encode_image(images, type, index) for index in range(batch_size)]

def write_image(image_data, filename):
    with open(filename, "wb") as image_file:
        image_file.write(image_data)

def estimate_percentile(im):
    mean = tf.reduce_mean(im)
    stdev = tf.sqrt(tf.reduce_mean((im - mean) ** 2.0))
    return mean + 1.645 * stdev

def tf_percentile(images):
    min = tf.reduce_min(tf.log(1.0 + images))
    max = tf.reduce_max(tf.log(1.0 + images))
    histogram = tf.histogram_fixed_width(tf.reshape(images, [-1]), [min, max])
    values = tf.linspace(min, max, 100)
    csum = tf.cumsum(histogram)
    csum_float = tf.cast(csum, tf.float32) / tf.cast(tf.size(csum), tf.float32)
    argmin_index = tf.cast(tf.argmin((csum_float - 0.95) ** 2.0, axis = 0), tf.int32)
    return tf.exp(values[argmin_index]) - 1.0

def tf_normalize(depth):
    disparity = 1.0 / (depth + 1e-6)
    normalized_disparity = disparity / (tf_percentile(disparity) + 1e-6)
    return tf.clip_by_value(normalized_disparity, 0.0, 1.0)
    
def gamma(images):
    A = 0.35
    b = 40.0
    n = tf.log(A) / tf.log(1.0 / b)
    return A * tf.clip_by_value(images, 0.0, b) ** n

def gray2rgb(im, cmap_name = "inferno", quantization_levels = 4096):
    batch_size = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    
    scale = float(quantization_levels) - 1.0
    
    im_clip = tf.clip_by_value(im * scale, 0.0, scale)
    im_flat = tf.reshape(tf.cast(im_clip, tf.int32), [-1])
    cmap = tf.constant(plt.get_cmap(cmap_name, quantization_levels).colors.astype(np.float32))
    rgb_im_flat = tf.gather(cmap, im_flat)
    rgb_im = tf.reshape(rgb_im_flat, [batch_size, height, width, 4])
    
    return rgb_im[:, :, :, 0:3]

def normalize_depth(depth):
    gamma = 1.8
    depth = 0.9 + tf.log(1.0 + depth ** gamma)
    #depth = tf.clip_by_value(depth ** (1.0 / gamma), 0.25, 40.0)
    #depth = tf.clip_by_value(depth, 0.15, 80.0)
    depth = 1.0 / (depth + 1e-6)
    #max = tf.reduce_max(depth, [1, 2], keep_dims = True)
    #min = tf.reduce_min(depth, [1, 2], keep_dims = True)
    #depth = (depth - min) / (max - min)
    return gray2rgb(depth)
    #return gray2rgb(tf_normalize(tf.clip_by_value(depth, 0.1, 40.0)))
    #return gray2rgb((tf.log(1.8) / (tf.log(1.0 + tf.clip_by_value(depth, 0.1, 10.0)) + 1e-6)) ** 1.6)
    #return gray2rgb(gamma(depth))

def normalize_disparity(disparity):
    return gray2rgb(3.0 * disparity)

def normalize(images):
    min = tf.reduce_min(images, [1, 2], keep_dims = True)
    max = tf.reduce_max(images, [1, 2], keep_dims = True)
    return (images - min) / (max - min)

def restore(images, target_height):
    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    channels = tf.shape(images)[3]

    th = tf.cast((target_height - height) / 2.0, tf.int32)
    bh = target_height - (height + th)

    tz = tf.zeros([batch_size, th, width, channels], dtype = tf.float32)
    bz = tf.zeros([batch_size, bh, width, channels], dtype = tf.float32)
    return tf.concat([tz, images, bz], 2)

def write_pc(pc, filename):
    num_points = pc.shape[0]
    with open(filename, "w") as pc_file:
        pc_file.write(str(num_points))

        for point_index in range(num_points):
            x = pc[point_index, 0]
            y = pc[point_index, 1]
            z = pc[point_index, 2]
            r = pc[point_index, 3]
            g = pc[point_index, 4]
            b = pc[point_index, 5]
            pc_file.write("\n{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(x, y, z, r, g, b))
