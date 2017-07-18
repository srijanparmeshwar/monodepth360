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
        # Shape constants.
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

        # Scale indices from [0, 1] to [0, width - 1] or [0, height - 1]
        x = x * (width_f - 1)
        y = y * (height_f - 1)

        # Do sampling.
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
        # channels dimension.
        im_flat = tf.reshape(input_images, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, "float32")
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # Finally calculate interpolated values.
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

def transform(input_images, x_t, y_t, x_offset, y_offset):
    with tf.variable_scope("transform"):
        batch_size   = tf.shape(input_images)[0]
        num_channels = tf.shape(input_images)[3]

        out_height = tf.shape(x_t)[0]
        out_width = tf.shape(x_t)[1]

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        x_t_flat = tf.tile(x_t_flat, tf.stack([batch_size, 1]))
        y_t_flat = tf.tile(y_t_flat, tf.stack([batch_size, 1]))

        x_t_flat = tf.reshape(x_t_flat, [-1])
        y_t_flat = tf.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat + tf.reshape(x_offset, [-1])
        y_t_flat = y_t_flat + tf.reshape(y_offset, [-1])

        input_transformed = interpolate(input_images, x_t_flat, y_t_flat, [out_height, out_width])

        output = tf.reshape(
            input_transformed, tf.stack([batch_size, out_height, out_width, num_channels]))
        return output

def uv_grid(shape):
    u, v = tf.meshgrid(tf.linspace(0.0, 1.0, shape[1]), tf.linspace(0.0, 1.0, shape[0]))
    return u, v

def bilinear_sample(input_images, x_t = None, y_t = None, x_offset = 0.0, y_offset = 0.0, name = "bilinear_sampler", **kwargs):
    with tf.variable_scope(name):
        height       = tf.shape(input_images)[1]
        width        = tf.shape(input_images)[2]

        if x_t is None and y_t is None:
            x_t, y_t = uv_grid([height, width])

        return transform(input_images, x_t, y_t, x_offset, y_offset)
