# Modifications 2017 Srijan Parmeshwar.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 Modifications Clement Godard.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf

def repeat(x, n_repeats):
    with tf.variable_scope("repeat"):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, "int32")
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])


def interpolate(input_images, x, y, out_size):
    with tf.variable_scope("interpolate"):
        # constants
        num_batch = tf.shape(input_images)[0]
        height = tf.shape(input_images)[1]
        width = tf.shape(input_images)[2]
        channels = tf.shape(input_images)[3]

        x = tf.cast(x, "float32")
        y = tf.cast(y, "float32")
        height_f = tf.cast(height, "float32")
        width_f = tf.cast(width, "float32")
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype="int32")
        max_y = tf.cast(tf.shape(input_images)[1] - 1, "int32")
        max_x = tf.cast(tf.shape(input_images)[2] - 1, "int32")

        # Scale indices from [0, 1] to [0, width/height]
        x = (x) * (width_f)
        y = (y) * (height_f)

        # Do sampling
        x0 = tf.cast(tf.floor(x), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), "int32")
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(input_images, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, "float32")
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # Finally calculate interpolated values
        x0_f = tf.cast(x0, "float32")
        x1_f = tf.cast(x1, "float32")
        y0_f = tf.cast(y0, "float32")
        y1_f = tf.cast(y1, "float32")
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output


# def repeat(x, n_repeats):
#     with tf.variable_scope("repeat"):
#         rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
#         return tf.reshape(rep, [-1])
#
#
# def interpolate(input_images, x, y, wrap_mode):
#     with tf.variable_scope("interpolate"):
#
#         batch_size   = tf.shape(input_images)[0]
#         height       = tf.shape(input_images)[1]
#         width        = tf.shape(input_images)[2]
#         num_channels = tf.shape(input_images)[3]
#
#         width_f  = tf.cast(width,  tf.float32)
#
#         # handle both texture border types
#         _edge_size = 0
#         if wrap_mode == "border":
#             _edge_size = 1
#             input_images = tf.pad(input_images, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
#             x = x + _edge_size
#             y = y + _edge_size
#         elif wrap_mode == "edge":
#             _edge_size = 0
#         else:
#             return None
#
#         x = tf.clip_by_value(x, 0.0, width_f - 1 + 2 * _edge_size)
#
#         x0_f = tf.floor(x)
#         y0_f = tf.floor(y)
#         x1_f = x0_f + 1
#
#         x0 = tf.cast(x0_f, tf.int32)
#         y0 = tf.cast(y0_f, tf.int32)
#         x1 = tf.cast(tf.minimum(x1_f, width_f - 1 + 2 * _edge_size), tf.int32)
#
#         dim2 = (width + 2 * _edge_size)
#         dim1 = (width + 2 * _edge_size) * (height + 2 * _edge_size)
#         base = repeat(tf.range(batch_size) * dim1, height * width)
#         base_y0 = base + y0 * dim2
#         idx_l = base_y0 + x0
#         idx_r = base_y0 + x1
#
#         im_flat = tf.reshape(input_images, tf.stack([-1, num_channels]))
#
#         pix_l = tf.gather(im_flat, idx_l)
#         pix_r = tf.gather(im_flat, idx_r)
#
#         weight_l = tf.expand_dims(x1_f - x, 1)
#         weight_r = tf.expand_dims(x - x0_f, 1)
#
#         return weight_l * pix_l + weight_r * pix_r


def transform(input_images, x_t, y_t, x_offset, y_offset):
    with tf.variable_scope("transform"):
        batch_size   = tf.shape(input_images)[0]
        height       = tf.shape(input_images)[1]
        width        = tf.shape(input_images)[2]
        num_channels = tf.shape(input_images)[3]

        height_f = tf.cast(height, tf.float32)
        width_f  = tf.cast(width,  tf.float32)

        out_height = tf.shape(x_t)[0]
        out_width = tf.shape(x_t)[1]

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        x_t_flat = tf.tile(x_t_flat, tf.stack([batch_size, 1]))
        y_t_flat = tf.tile(y_t_flat, tf.stack([batch_size, 1]))

        x_t_flat = tf.reshape(x_t_flat, [-1])
        y_t_flat = tf.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * width_f
        y_t_flat = y_t_flat + tf.reshape(y_offset, [-1]) * height_f

        input_transformed = interpolate(input_images, x_t_flat, y_t_flat, tf.shape(x_t))

        output = tf.reshape(
            input_transformed, tf.stack([batch_size, out_height, out_width, num_channels]))
        return output

def bilinear_sample(input_images, x_t = None, y_t = None, x_offset = 0.0, y_offset = 0.0, name = "bilinear_sampler", **kwargs):
    with tf.variable_scope(name):
        height       = tf.shape(input_images)[1]
        width        = tf.shape(input_images)[2]

        height_f = tf.cast(height, tf.float32)
        width_f  = tf.cast(width,  tf.float32)

        if x_t is None and y_t is None:
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, width_f - 1.0, width),
                                   tf.linspace(0.0, height_f - 1.0, height))

        return transform(input_images, x_t, y_t, x_offset, y_offset)
