from bilinear_sampler import bilinear_sample

import numpy as np
import tensorflow as tf

#  Taken from asos-ben implementation at https://github.com/tensorflow/tensorflow/issues/6095
def atan2(x, y, epsilon = 1.0e-12):
    """
    A hack until the tf developers implement a function that can find the angle from an x and y co-
    ordinate.
    :param x:
    :param epsilon:
    :return:
    """
    # Add a small number to all zeros, to avoid division by zero:
    x = tf.where(tf.equal(x, 0.0), x + epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y + epsilon, y)

    angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.greater_equal(y, 0.0)), tf.atan(y / x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.less(y, 0.0)), tf.atan(y / x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.greater(y, 0.0)), 0.5 * np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.less(y, 0.0)), -0.5 * np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)), tf.zeros_like(x), angle)
    return angle

face_map = [
    "front",
    "back",
    "left",
    "right",
    "up",
    "down"
]

def lat_long_grid(shape, epsilon = 1.0e-12):
    return tf.meshgrid(tf.linspace(-np.pi, np.pi, shape[1]),
                       tf.linspace(-np.pi / 2.0 + epsilon, np.pi / 2.0 - epsilon, shape[0]))

def uv_grid(shape):
    return tf.meshgrid(tf.linspace(-0.5, 0.5, shape[1]),
                       tf.linspace(-0.5, 0.5, shape[0]))

def xyz_grid(shape, face = "front"):
    a, b = tf.meshgrid(tf.linspace(-1.0, 1.0, shape[1]),
                       tf.linspace(-1.0, 1.0, shape[0]))
    c = tf.constant(1.0, dtype = tf.float32, shape = shape)

    if face == "front":
        x = a
        y = -b
        z = c
    elif face == "back":
        x = -a
        y = -b
        z = -c
    elif face == "left":
        x = -c
        y = -b
        z = a
    elif face == "right":
        x = c
        y = -b
        z = -a
    elif face == "up":
        x = a
        y = c
        z = b
    else:
        x = a
        y = -c
        z = -b

    return x, y, z

def xyz_to_lat_long(x, y, z):
    S = -atan2(x, z)
    T = atan2(y, tf.sqrt(x ** 2.0 + z ** 2.0))
    return S, T

def lat_long_to_xyz(S, T):
    x = tf.cos(T) * tf.sin(S)
    y = tf.sin(T)
    z = tf.cos(T) * tf.cos(S)
    return x, y, z

def backproject(S, T, depth):
    x = depth * tf.sin(S)
    y = depth * tf.tan(T)
    z = depth * tf.cos(S)
    return x, y, z

def lat_long_to_cube_uv(S, T):
    x = tf.cos(T) * tf.sin(S)
    y = tf.sin(T)
    z = tf.cos(T) * tf.cos(S)

    argmax = tf.argmax(tf.abs([x, y, z]), axis = 0)
    max = tf.reduce_max(tf.abs([x, y, z]), axis = 0)

    front_check = tf.logical_and(
        tf.equal(argmax, 2),
        tf.greater_equal(z, 0.0)
    )
    back_check = tf.logical_and(
        tf.equal(argmax, 2),
        tf.less(z, 0.0)
    )
    left_check = tf.logical_and(
        tf.equal(argmax, 0),
        tf.less(x, 0.0)
    )
    right_check = tf.logical_and(
        tf.equal(argmax, 0),
        tf.greater_equal(x, 0.0)
    )
    up_check = tf.logical_and(
        tf.equal(argmax, 1),
        tf.less(y, 0.0)
    )
    down_check = tf.logical_and(
        tf.equal(argmax, 1),
        tf.greater_equal(y, 0.0)
    )

    x = x / max
    y = y / max
    z = z / max

    u = tf.where(front_check, 0.5 + x / 2.0, tf.zeros_like(x))
    u = tf.where(back_check, 1.0 + (0.5 - x / 2.0), u)
    u = tf.where(left_check, 2.0 + (0.5 + z / 2.0), u)
    u = tf.where(right_check, 3.0 + (0.5 - z / 2.0), u)
    u = tf.where(up_check, 4.0 + (0.5 + x / 2.0), u)
    u = tf.where(down_check, 5.0 + (0.5 + x / 2.0), u)
    u = u / 6.0

    v = tf.where(front_check, (1.0 + y) / 2.0, tf.zeros_like(y))
    v = tf.where(back_check, (1.0 + y) / 2.0, v)
    v = tf.where(left_check, (1.0 + y) / 2.0, v)
    v = tf.where(right_check, (1.0 + y) / 2.0, v)
    v = tf.where(up_check, (1.0 + z) / 2.0, v)
    v = tf.where(down_check, (1.0 - z) / 2.0, v)

    return u, v

def mod(x, c):
    x = tf.where(tf.less(x, 0.0), x + c, x)
    x = tf.where(tf.greater_equal(x, c), x - c, x)
    return x

def lat_long_to_equirectangular_uv(S, T):
    u = tf.mod(S / (2.0 * np.pi) - 0.25, 1.0)
    v = tf.mod(T / np.pi, 1.0)
    return u, v

def project_face(input_images, face, cubic_shape):
    x, y, z = xyz_grid(cubic_shape, face)
    S, T = xyz_to_lat_long(x, y, z)
    u, v = lat_long_to_equirectangular_uv(S, T)
    return bilinear_sample(input_images, u, v)

def stack_faces(faces):
    return tf.concat(faces, 2)

def equirectangular_to_cubic(input_images, cubic_shape):
    return [project_face(input_images, face, cubic_shape) for face in face_map]

def cubic_to_equirectangular(input_images, equirectangular_shape):
    stacked_faces = stack_faces(input_images)
    S, T = lat_long_grid(equirectangular_shape)
    u, v = lat_long_to_cube_uv(S, T)
    return bilinear_sample(stacked_faces, u, v)
