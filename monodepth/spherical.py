from bilinear_sampler import bilinear_sample

import numpy as np
import tensorflow as tf

#  Taken from asos-ben implementation at https://github.com/tensorflow/tensorflow/issues/6095
# If using a later TensorFlow version, can change to tf.atan2.
def atan2(x, y, epsilon = 1.0e-12):
    """
    A hack until the TensorFlow developers implement a function that can find the angle from an x and y co-
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

# List of faces for consistent ordering.
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

# Restricted rotations of (a, b, c) to (x, y, z), implemented using
# permutations and negations.
def switch_face(a, b, c, face = "front"):
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

def xyz_grid(shape, face = "front"):
    a, b = tf.meshgrid(tf.linspace(-1.0, 1.0, shape[1]),
                       tf.linspace(-1.0, 1.0, shape[0]))
    c = tf.constant(1.0, dtype = tf.float32, shape = shape)

    return switch_face(a, b, c, face)

# Convert Cartesian coordinates (x, y, z) to latitude (T) and longitude (S).
def xyz_to_lat_long(x, y, z):
    S = - atan2(x, z)
    T = atan2(y, tf.sqrt(x ** 2.0 + z ** 2.0))
    return S, T

# Convert latitude (T) and longitude (S) to Cartesian coordinates (x, y, z).
def lat_long_to_xyz(S, T):
    x = tf.cos(T) * tf.sin(S)
    y = tf.sin(T)
    z = tf.cos(T) * tf.cos(S)
    return x, y, z

def backproject_cubic_depth(depth, shape, face):
    a, b = tf.meshgrid(tf.linspace(-1.0, 1.0, shape[2]),
                       tf.linspace(-1.0, 1.0, shape[1]))
    A = depth * tf.expand_dims(tf.tile(tf.expand_dims(a, 0), [shape[0], 1, 1]), 3)
    B = depth * tf.expand_dims(tf.tile(tf.expand_dims(b, 0), [shape[0], 1, 1]), 3)
    C = depth

    x, y, z = switch_face(A, B, C, face)

    return tf.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)

def backproject_cubic(depth, shape, face):
    a, b = tf.meshgrid(tf.linspace(-1.0, 1.0, shape[2]),
                       tf.linspace(-1.0, 1.0, shape[1]))
    A = depth * tf.expand_dims(tf.tile(tf.expand_dims(a, 0), [shape[0], 1, 1]), 3)
    B = depth * tf.expand_dims(tf.tile(tf.expand_dims(b, 0), [shape[0], 1, 1]), 3)
    C = depth

    x, y, z = switch_face(A, B, C, face)

    return tf.sqrt(x ** 2.0 + z ** 2.0)

def backproject_rectilinear(depth, K, shape, face):
    u, v = tf.meshgrid(tf.linspace(-1.0, 1.0, shape[2]),
                       tf.linspace(-1.0, 1.0, shape[1]))

    u = tf.expand_dims(tf.tile(tf.expand_dims(u, 0), [shape[0], 1, 1]), 3)
    v = tf.expand_dims(tf.tile(tf.expand_dims(v, 0), [shape[0], 1, 1]), 3)

    A = (u - K[2]) * depth / K[0]
    B = (v - K[3]) * depth / K[1]
    C = depth

    x, y, z = switch_face(A, B, C, face)

    return tf.sqrt(x ** 2.0 + z ** 2.0)

def backproject(S, T, depth):
    # Convert to Cartesian for modified depth input.
    # depth = sqrt(x^2 + z^2).
    x = depth * tf.sin(S)
    y = depth * tf.tan(T)
    z = depth * tf.cos(S)
    return x, y, z

def rectilinear_xyz(K, shape, face = "front"):
    u, v = tf.meshgrid(tf.linspace(-1.0, 1.0, shape[1]),
                       tf.linspace(-1.0, 1.0, shape[0]))
    # X = (u - c_x) * z / f_x
    # Y = (v - c_y) * z / f_y
    a = (u - K[2]) / K[0]
    b = (v - K[3]) / K[1]
    c = tf.ones([shape[1], shape[0]], dtype = tf.float32)

    return switch_face(a, b, c, face)

def lat_long_to_rectilinear_uv(K, S, T):
    # Convert to Cartesian.
    x = tf.cos(T) * tf.sin(S)
    y = tf.sin(T)
    z = tf.cos(T) * tf.cos(S)

    argmax = tf.argmax(tf.abs([x, y, z]), axis = 0)

    # Check which face the ray lies on.
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
    
    def project_u(x, y, z, offset):
        return offset + 0.5 + (K[2] + K[0] * x / z) / 2.0

    def project_v(x, y, z):
        return 0.5 + (K[3] + K[1] * y / z) / 2.0

    # Calculate UV coordinates.
    u = tf.where(front_check, project_u(x, y, z, 0.0), tf.zeros_like(x))
    u = tf.where(back_check, project_u(x, -y, z, 1.0), u)
    u = tf.where(left_check, project_u(z, y, -x, 2.0), u)
    u = tf.where(right_check, project_u(-z, y, x, 3.0), u)
    u = tf.where(up_check, project_u(x, z, -y, 4.0), u)
    u = tf.where(down_check, project_u(x, -z, y, 5.0), u)
    u = u / 6.0
    
    v = tf.where(front_check, project_v(x, y, z), tf.zeros_like(y))
    v = tf.where(back_check, project_v(x, -y, z), v)
    v = tf.where(left_check, project_v(z, y, -x), v)
    v = tf.where(right_check, project_v(-z, y, x), v)
    v = tf.where(up_check, project_v(x, z, -y), v)
    v = tf.where(down_check, project_v(x, -z, y), v)

    return u, v

def lat_long_to_cube_uv(S, T):
    # Convert to Cartesian.
    x = tf.cos(T) * tf.sin(S)
    y = tf.sin(T)
    z = tf.cos(T) * tf.cos(S)

    argmax = tf.argmax(tf.abs([x, y, z]), axis = 0)
    max = tf.reduce_max(tf.abs([x, y, z]), axis = 0)

    # Check which face the ray lies on.
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

    # Normalize coordinates.
    x = x / max
    y = y / max
    z = z / max

    # Calculate UV coordinates.
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

def lat_long_to_equirectangular_uv(S, T):
    # Convert latitude and longitude to UV coordinates
    # on an equirectangular plane.
    u = tf.mod(S / (2.0 * np.pi) - 0.25, 1.0)
    v = tf.mod(T / np.pi, 1.0)
    return u, v

# General rotation function given angles in (x, y, z) axes.
def rotate(input_images, rx, ry, rz):
    # Create constants.
    batch_size = tf.shape(input_images)[0]
    height = tf.shape(input_images)[1]
    width = tf.shape(input_images)[2]
    batch_zero = tf.tile(tf.constant(0.0, shape = [1]), [batch_size])
    batch_one = tf.tile(tf.constant(1.0, shape = [1]), [batch_size])

    # Function to parse Python lists as TensorFlow matrices.
    def tf_batch_matrix(matrix):
        return tf.transpose(tf.stack(matrix), [2, 0, 1])

    # Convert to Cartesian.
    S, T = lat_long_grid([height, width])
    x, y, z = lat_long_to_xyz(S, T)
    X = tf.tile(tf.expand_dims(tf.reshape(tf.stack([x, y, z]), [3, height * width]), 0), [batch_size, 1, 1])

    # Construct rotation matrices (for inverse warp).
    rx = - rx
    ry = - ry
    rz = - rz
    R = tf_batch_matrix([
            [tf.cos(rz), - tf.sin(rz), batch_zero],
            [tf.sin(rz), tf.cos(rz), batch_zero],
            [batch_zero, batch_zero, batch_one]
        ])
    R = tf.matmul(
            tf_batch_matrix([
                [tf.cos(ry), batch_zero, tf.sin(ry)],
                [batch_zero, batch_one, batch_zero],
                [- tf.sin(ry), batch_zero, tf.cos(ry)]
            ]),
        R)
    R = tf.matmul(
            tf_batch_matrix([
                [batch_one, batch_zero, batch_zero],
                [batch_zero, tf.cos(rx), - tf.sin(rx)],
                [batch_zero, tf.sin(rx), tf.cos(rx)]
            ]),
        R)

    # Rotate coordinates.
    X_rotated = tf.reshape(tf.matmul(R, X), [batch_size, 3, height, width])

    # Convert back to equirectangular UV.
    S_rotated, T_rotated = xyz_to_lat_long(X_rotated[:, 0, :, :], X_rotated[:, 1, :, :], X_rotated[:, 2, :, :])
    u, v = lat_long_to_equirectangular_uv(S_rotated, T_rotated)
    
    return bilinear_sample(input_images, x_t = tf.zeros_like(u[0]), y_t = tf.zeros_like(v[0]), x_offset = u, y_offset = 1.0 - v)

def fast_rotate(input_image, dx = 0, dy = 0):
    # Basic rotations (constant disparities) for equirectangular
    # images. For image augmentations (y-axis rotations), this method is preferable compared
    # to the more general rotation function.
    height = tf.shape(input_image)[0]
    width = tf.shape(input_image)[1]

    # Shift coordinate grid for inverse warp.
    ix, iy = tf.meshgrid(tf.range(width), tf.range(height))
    ox = tf.mod(ix - dx, width)
    oy = tf.mod(iy - dy, height)
    indices = tf.stack([oy, ox], 2)

    # Perform exact sampling (as we are using integer coordinates).
    return tf.gather_nd(input_image, indices)

# Project equirectangular image onto a cube face.
def project_face(input_images, face, cubic_shape):
    x, y, z = xyz_grid(cubic_shape, face)
    S, T = xyz_to_lat_long(x, y, z)
    u, v = lat_long_to_equirectangular_uv(S, T)
    return bilinear_sample(input_images, u, v)

# Project equirectangular image into rectilinear camera image using given intrinsics K.
def project_rectilinear(input_images, K, face, face_shape):
    x, y, z = rectilinear_xyz(K, face_shape, face)
    S, T = xyz_to_lat_long(x, y, z)
    u, v = lat_long_to_equirectangular_uv(S, T)
    return bilinear_sample(input_images, u, v)

def stack_faces(faces):
    # Stack faces horizontally on image plane.
    # Used for bilinear sampling on from multiple images - for cube map and rectilinear projections.
    return tf.concat(faces, 2)

# Convert spherical depth to distance.
def perpendicular_to_distance(depths):
    batch_size = tf.shape(depths)[0]
    height = tf.shape(depths)[1]
    width = tf.shape(depths)[2]

    S, T = lat_long_grid([height, width])
    S_grids = tf.tile(tf.reshape(S, [1, height, width, 1]), [batch_size, 1, 1, 1])
    T_grids = tf.tile(tf.reshape(T, [1, height, width, 1]), [batch_size, 1, 1, 1])

    x, y, z = backproject(S_grids, T_grids, depths)

    return tf.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)

# Backproject equirectangular image to a point cloud from given depth values.
def equirectangular_to_pc(input_images, depths):
    batch_size = tf.shape(input_images)[0]
    height = tf.shape(input_images)[1]
    width = tf.shape(input_images)[2]

    S, T = lat_long_grid([height, width])
    S_grids = tf.tile(tf.reshape(S, [1, height, width, 1]), [batch_size, 1, 1, 1])
    T_grids = tf.tile(tf.reshape(T, [1, height, width, 1]), [batch_size, 1, 1, 1])

    X = tf.stack(backproject(S_grids, T_grids, depths), 3)
    pc = tf.stack([tf.squeeze(X, 4), input_images], 3)

    return tf.reshape(pc, [batch_size, -1, 6])

def equirectangular_to_cubic(input_images, cubic_shape):
    return [project_face(input_images, face, cubic_shape) for face in face_map]

def equirectangular_to_rectilinear(input_images, K, face_shape):
    return [project_rectilinear(input_images, K, face, face_shape) for face in face_map]

def cubic_to_equirectangular(input_images, equirectangular_shape):
    stacked_faces = stack_faces(input_images)
    S, T = lat_long_grid(equirectangular_shape)
    u, v = lat_long_to_cube_uv(S, T)
    return bilinear_sample(stacked_faces, u, v)

def rectilinear_to_equirectangular(input_images, K, equirectangular_shape):
    stacked_faces = stack_faces(input_images)
    S, T = lat_long_grid(equirectangular_shape)
    u, v = lat_long_to_rectilinear_uv(K, S, T)
    return bilinear_sample(stacked_faces, u, v)
