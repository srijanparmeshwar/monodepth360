# Modifications Srijan Parmeshwar 2017.
# Copyright UCL Business plc 2017. Patent Pending. All bottoms reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *
from collections import namedtuple
from image_utils import normalize_depth
from image_utils import restore
from spherical import *

monodepth_parameters = namedtuple('parameters',
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'projection,'
                        'baseline,'
                        'output_mode,'
                        'use_deconv, '
                        'alpha_image_loss, '
                        'smoothness_loss_weight, '
                        'dual_loss, '
                        'crop, '
                        'test_crop, '
                        'dropout, '
                        'noise, '
                        'tb_loss_weight, '
                        'full_summary')

class MonodepthModel(object):
    """Monodepth model"""

    def __init__(self, params, mode, top, bottom, reuse_variables = None, model_index = 0):
        self.params = params
        self.mode = mode
        self.top = top
        self.bottom = bottom
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        if self.params.projection == 'rectilinear':
            self.rectilinear_net()
        elif self.params.projection == 'equirectangular':
            self.equirectangular_net()
        else:
            print("Projection {} did not match either rectilinear or equirectangular. Defaulting to equirectangular.".format(self.params.projection))
            self.equirectangular_net()

        self.build_depths_and_disparities()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()     

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h / ratio
            nw = w / ratio
            scaled_imgs.append(tf.image.resize_area(img, tf.cast([nh, nw], tf.int32)))
        return scaled_imgs

    def pyramid_shapes(self, shape, num_scales):
        shapes = [shape]
        h = shape[0]
        w = shape[1]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h / ratio
            nw = w / ratio
            shapes.append([nh, nw])
        return tf.cast(shapes, tf.int32)
    
    def expand_grids(self, S, T, batch_size):
        S_grids = tf.expand_dims(tf.tile(tf.expand_dims(S, 0), [batch_size, 1, 1]), 3)
        T_grids = tf.expand_dims(tf.tile(tf.expand_dims(T, 0), [batch_size, 1, 1]), 3)
        return S_grids, T_grids

    def attenuate_rectilinear(self, K, disparity, position):
        S, T = lat_long_grid([tf.shape(disparity)[1], tf.shape(disparity)[2]])
        _, T_grids = self.expand_grids(S, -T, tf.shape(disparity)[0])
        if position == "top":
            attenuated_disparity = (1.0 / np.pi) * (tf.atan(disparity / K[1] + tf.tan(T_grids)) - T_grids)
        else:
            attenuated_disparity = (1.0 / np.pi) * (T_grids - tf.atan(tf.tan(T_grids) - disparity / K[1]))
        return tf.clip_by_value(tf.where(tf.is_finite(attenuated_disparity), attenuated_disparity, tf.zeros_like(attenuated_disparity)), 1e-6, 0.75)
    
    def attenuate_equirectangular(self, disparity, position):
        S, T = lat_long_grid([tf.shape(disparity)[1], tf.shape(disparity)[2]])
        _, T_grids = self.expand_grids(S, -T, tf.shape(disparity)[0])
        if position == "top":
            attenuated_disparity = (1.0 / np.pi) * (tf.atan(tf.tan(np.pi * disparity) + tf.tan(T_grids)) - T_grids)
        else:
            attenuated_disparity = (1.0 / np.pi) * (T_grids - tf.atan(tf.tan(T_grids) - tf.tan(np.pi * disparity)))
        return tf.clip_by_value(tf.where(tf.is_finite(attenuated_disparity), attenuated_disparity, tf.zeros_like(attenuated_disparity)), 1e-6, 0.75)
    
    def rectilinear_disparity_to_depth(self, disparity, K, face, epsilon = 1e-6):
        rectilinear_depth = K[1] * self.params.baseline / (disparity + epsilon)
        return backproject_rectilinear(rectilinear_depth, K, tf.shape(disparity), face)

    def equirectangular_disparity_to_depth(self, disparity, epsilon = 1e-6):
        return self.params.baseline / (disparity + epsilon)

    def disparity_to_depth(self, disparity, position, epsilon = 1e-6):
        baseline_distance = self.params.baseline
        S, T = lat_long_grid([tf.shape(disparity)[1], tf.shape(disparity)[2]])
        _, T_grids = self.expand_grids(S, -T, tf.shape(disparity)[0])
        if position == "top":
            t1 = tf.tan(T_grids)
            t2 = tf.tan(T_grids + np.pi * disparity)
        else:
            t1 = tf.tan(T_grids)
            t2 = tf.tan(T_grids - np.pi * disparity)
        return baseline_distance / (tf.abs(t2 - t1) + epsilon)

    def depth_to_disparity(self, depth, position):
        baseline_distance = self.params.baseline
        S, T = lat_long_grid([tf.shape(depth)[1], tf.shape(depth)[2]])
        _, T_grids = self.expand_grids(S, T, tf.shape(depth)[0])
        if position == "top":
            return self.disparity_scale * (np.pi / 2.0 - atan2(baseline_distance * depth, (1.0 + tf.tan(-T_grids) ** 2.0) * (depth ** 2.0) + baseline_distance * depth * tf.tan(-T_grids)))
        else:
            return self.disparity_scale * (atan2(baseline_distance * depth, (1.0 + tf.tan(-T_grids) ** 2.0) * (depth ** 2.0) - baseline_distance * depth * tf.tan(-T_grids)) - np.pi / 2.0)

    def generate_image_top(self, img, disp):
        return bilinear_sample(img, x_t = None, y_t = None, x_offset = 0.0, y_offset = -disp)

    def generate_image_bottom(self, img, disp):
        return bilinear_sample(img, x_t = None, y_t = None, x_offset = 0.0, y_offset = disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_smoothness(self, input_images, pyramid):
        gradients_x = [self.gradient_x(tf.abs(i)) for i in input_images]
        gradients_y = [self.gradient_y(tf.abs(i)) for i in input_images]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims = True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims = True)) for g in image_gradients_y]

        smoothness_x = [gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disparity(self, x, scale):
        disparity = scale * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disparity

    def get_depth(self, x, epsilon = 1e-1):
        depth = self.conv(x, 2, 3, 1, tf.nn.relu) + epsilon
        return depth

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn = tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn = activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]


    def random_noise(self, image):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]

        # Randomly shift gamma.
        random_gamma = tf.random_uniform([batch_size, 1, 1, 1], 0.8, 1.2)
        gamma_image = tf.tile(random_gamma, [1, height, width, 3])
        image_aug = image ** gamma_image

        # Randomly shift brightness.
        random_brightness = tf.random_uniform([batch_size, 1, 1, 1], 0.5, 2.0)
        brightness_image = tf.tile(random_brightness, [1, height, width, 3])
        image_aug = image_aug * brightness_image

        # Randomly shift color.
        random_colors = tf.random_uniform([batch_size, 1, 1, 3], 0.8, 1.2)
        color_image = tf.tile(random_colors, [1, height, width, 1])
        image_aug *= color_image

        # Saturate.
        image_aug = tf.clip_by_value(image_aug, 0, 1)

        return image_aug

    def noisy_resnet50(self, input, scope):
        iterations = 8
        outputs1 = []
        outputs2 = []
        outputs3 = []
        outputs4 = []

        noisy_input = [self.random_noise(input) for _ in range(iterations)]

        for iteration in range(iterations):
            output1, output2, output3, output4 = self.resnet50(noisy_input)
            outputs1.append(output1)
            outputs2.append(output2)
            outputs3.append(output3)
            outputs4.append(output4)
            if iteration + 1 < iterations:
                scope.reuse_variables()

        mean1 = tf.add_n(outputs1) / iterations
        mean2 = tf.add_n(outputs2) / iterations
        mean3 = tf.add_n(outputs3) / iterations
        mean4 = tf.add_n(outputs4) / iterations

        variance1 = tf.add_n([(output - mean1) ** 2.0 for output in outputs1]) / (iterations - 1)
        variance2 = tf.add_n([(output - mean2) ** 2.0 for output in outputs2]) / (iterations - 1)
        variance3 = tf.add_n([(output - mean3) ** 2.0 for output in outputs3]) / (iterations - 1)
        variance4 = tf.add_n([(output - mean4) ** 2.0 for output in outputs4]) / (iterations - 1)

        with tf.variable_scope("confidence"):
            self.confidence1 = variance1
            self.confidence2 = variance2
            self.confidence3 = variance3
            self.confidence4 = variance4

        return mean1, mean2, mean3, mean4

    def dropout_resnet50(self, input, scope):
        iterations = 8
        outputs1 = []
        outputs2 = []
        outputs3 = []
        outputs4 = []

        for iteration in range(iterations):
            output1, output2, output3, output4 = self.resnet50(input, True)
            outputs1.append(output1)
            outputs2.append(output2)
            outputs3.append(output3)
            outputs4.append(output4)
            if iteration + 1 < iterations:
                scope.reuse_variables()

        mean1 = tf.add_n(outputs1) / iterations
        mean2 = tf.add_n(outputs2) / iterations
        mean3 = tf.add_n(outputs3) / iterations
        mean4 = tf.add_n(outputs4) / iterations

        variance1 = tf.add_n([(output - mean1) ** 2.0 for output in outputs1]) / (iterations - 1)
        variance2 = tf.add_n([(output - mean2) ** 2.0 for output in outputs2]) / (iterations - 1)
        variance3 = tf.add_n([(output - mean3) ** 2.0 for output in outputs3]) / (iterations - 1)
        variance4 = tf.add_n([(output - mean4) ** 2.0 for output in outputs4]) / (iterations - 1)

        with tf.variable_scope("confidence"):
            self.confidence1 = variance1
            self.confidence2 = variance2
            self.confidence3 = variance3
            self.confidence4 = variance4

        return mean1, mean2, mean3, mean4

    def resnet50(self, input, dropout = False):
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        if self.params.output_mode == "direct":
            get_layer = lambda x: self.get_disparity(x, 0.5)
        else:
            get_layer = lambda x: self.get_disparity(x, 0.5)
        #get_layer = lambda x: self.get_disparity(x, 0.5)

        dropout_rate = 0.5
        dropout_function = lambda x: tf.layers.dropout(inputs = x, rate = dropout_rate, training = dropout)

        with tf.variable_scope('encoder'):
            conv1 = conv(input, 64, 7, 2) # H/2  -   64D
            pool1 = self.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
            conv3 = dropout_function(conv3)
            conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D
            conv4 = dropout_function(conv4)
            conv5 = self.resblock(conv4,     512, 3) # H/64 - 2048D
            conv5 = dropout_function(conv5)

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            output4 = get_layer(iconv4)
            udepth4  = self.upsample_nn(output4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udepth4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            output3 = get_layer(iconv3)
            udepth3  = self.upsample_nn(output3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udepth3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            output2 = get_layer(iconv2)
            udepth2  = self.upsample_nn(output2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udepth2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            output1 = get_layer(iconv1)

            return output1, output2, output3, output4

    def equirectangular_net(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn = tf.nn.elu):
            with tf.variable_scope("model", reuse = self.reuse_variables) as scope:
                # Calculate pyramid for equirectangular top image.
                self.top_pyramid = self.scale_pyramid(self.top, 4)

                with tf.variable_scope("scaling"):
                    self.depth_scale = tf.constant(0.25, shape = [1])
                    self.disparity_scale = tf.get_variable("disparity_scale", shape = [1], trainable = False,
                                                           initializer = tf.constant_initializer(1.0 / np.pi))

                if self.params.dropout:
                    resnet50 = lambda x: self.dropout_resnet50(x, scope)
                elif self.params.noise:
                    resnet50 = lambda x: self.noisy_resnet50(x, scope)
                else:
                    resnet50 = lambda x: self.resnet50(x, False)

                if self.mode == 'train':
                    # Calculate pyramid for equirectangular bottom image.
                    self.bottom_pyramid = self.scale_pyramid(self.bottom, 4)

                if self.params.test_crop:
                    crop_height = int(self.params.height / 8)
                    output1, output2, output3, output4 = resnet50(self.top[:, crop_height:-crop_height, :, :])
                else:
                    output1, output2, output3, output4 = resnet50(self.top)
                outputs = [output1, output2, output3, output4]

                if self.params.test_crop:
                    outputs = [restore(output, self.params.height) for output in outputs]

                if self.params.output_mode == "indirect":
                    self.outputs = [self.equirectangular_disparity_to_depth(output) for output in outputs]
                elif self.params.output_mode == "direct":
                    self.outputs = outputs
                elif self.params.output_mode == "attenuate":
                    self.outputs = [tf.concat(
                            [self.attenuate_equirectangular(tf.expand_dims(output[:, :, :, 0], 3), "top"), self.attenuate_equirectangular(tf.expand_dims(output[:, :, :, 1], 3), "bottom")],
                            3
                        ) for output in outputs]

    def get_intrinsics(self):
        # Settings for overlap in rectilinear faces.
        #padding_scale = 1.5
        padding_scale = 2.0
        zoom = 1.0 / padding_scale
        # Intrinsic parameters from zoom level.
        K = [zoom, zoom, 0.0, 0.0]
    
        return K
    
    def rectilinear_net(self):
        K = self.get_intrinsics()
        batch_size = tf.shape(self.top)[0]
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn = tf.nn.elu):
            with tf.variable_scope("model", reuse = self.reuse_variables) as scope:
                # Calculate pyramid for equirectangular top image.
                self.top_pyramid = self.scale_pyramid(self.top, 4)
                square_size = self.params.height / 2

                # Convert top image into cubic format.
                self.top_faces = [tf.reshape(face,
                                             [batch_size, square_size, square_size, 3]) for face in
                                  equirectangular_to_rectilinear(self.top,
                                                                 K,
                                                           [square_size, square_size])]
                with tf.variable_scope("scaling"):
                    self.depth_scale = tf.constant(0.25, shape=[1])
                    self.disparity_scale = tf.get_variable("disparity_scale", shape = [1], trainable = False,
                                                           initializer = tf.constant_initializer(1.0 / np.pi))

                if self.params.dropout:
                    resnet50 = lambda x: self.dropout_resnet50(x, scope)
                elif self.params.noise:
                    resnet50 = lambda x: self.noisy_resnet50(x, scope)
                else:
                    resnet50 = lambda x: self.resnet50(x, False)

                if self.mode == 'train':
                    # Calculate pyramid for equirectangular bottom image.
                    self.bottom_pyramid = self.scale_pyramid(self.bottom, 4)

                # Calculate disparity and depth maps for each face direction individually.
                output_pyramids = [[] for _ in range(4)]
                pyramid_shapes = self.pyramid_shapes([self.params.height, self.params.width], 4)

                for face_index in range(6):
                    output1, output2, output3, output4 = resnet50(self.top_faces[face_index])
                    if face_index < 5:
                        scope.reuse_variables()

                    if self.params.output_mode == "indirect":
                        output_pyramids[0].append(self.rectilinear_disparity_to_depth(output1, K, face_map[face_index]))
                        output_pyramids[1].append(self.rectilinear_disparity_to_depth(output2, K, face_map[face_index]))
                        output_pyramids[2].append(self.rectilinear_disparity_to_depth(output3, K, face_map[face_index]))
                        output_pyramids[3].append(self.rectilinear_disparity_to_depth(output4, K, face_map[face_index]))
                    elif self.params.output_mode == "direct" or self.params.output_mode == "attenuate":
                        output_pyramids[0].append(output1)
                        output_pyramids[1].append(output2)
                        output_pyramids[2].append(output3)
                        output_pyramids[3].append(output4)

                # Convert depth maps to equirectangular format.
                self.outputs = [
                    rectilinear_to_equirectangular(
                        output_pyramids[scale_index],
                        K,
                        pyramid_shapes[scale_index]
                    )
                    for scale_index in range(4)
                ]
                
                if self.params.output_mode == "attenuate":
                    self.outputs = [tf.concat(
                            [self.attenuate_rectilinear(K, tf.expand_dims(output[:, :, :, 0], 3), "top"), self.attenuate_rectilinear(K, tf.expand_dims(output[:, :, :, 1], 3), "bottom")],
                            3
                        ) for output in self.outputs]

    def build_depths_and_disparities(self):
        if self.params.output_mode == "direct" or self.params.output_mode == "attenuate":
            with tf.variable_scope('disparities'):
                self.disparity_top_est = [tf.expand_dims(output[:, :, :, 0], 3) for output in self.outputs]
                self.disparity_bottom_est = [tf.expand_dims(output[:, :, :, 1], 3) for output in self.outputs]

            with tf.variable_scope('depths'):
                self.depth_top_est = [self.disparity_to_depth(disparity, "top") for disparity in
                                      self.disparity_top_est]
                self.depth_bottom_est = [self.disparity_to_depth(disparity, "bottom") for disparity in
                                         self.disparity_bottom_est]

        elif self.params.output_mode == "indirect":
            # Store depth maps.
            with tf.variable_scope('depths'):
                self.depth_top_est = [tf.expand_dims(output[:, :, :, 0], 3) for output in self.outputs]
                self.depth_bottom_est = [tf.expand_dims(output[:, :, :, 1], 3) for output in self.outputs]

            # Store vertical disparities maps.
            with tf.variable_scope('disparities'):
                self.disparity_top_est = [self.depth_to_disparity(depth, "top") for depth in
                                          self.depth_top_est]
                self.disparity_bottom_est = [self.depth_to_disparity(depth, "bottom") for depth in
                                             self.depth_bottom_est]

    def build_outputs(self):
        # Generate bottom image.
        with tf.variable_scope('images'):
            self.bottom_est = [self.generate_image_bottom(self.top_pyramid[i], self.disparity_bottom_est[i]) for i in range(4)]

        if self.mode == 'test':
            return

        # Generate top image.
        with tf.variable_scope('images'):
            self.top_est  = [self.generate_image_top(self.bottom_pyramid[i], self.disparity_top_est[i])  for i in range(4)]

        # Top-bottom consistency.
        with tf.variable_scope('top-bottom'):
            if self.params.dual_loss:
                self.bottom_to_top_depth = [self.generate_image_top(tf.log(1.0 + tf.abs(self.depth_bottom_est[i])), self.disparity_top_est[i])  for i in range(4)]
                self.top_to_bottom_depth = [self.generate_image_bottom(tf.log(1.0 + tf.abs(self.depth_top_est[i])), self.disparity_bottom_est[i]) for i in range(4)]

            self.bottom_to_top_disparity = [self.generate_image_top(tf.abs(self.disparity_bottom_est[i]), self.disparity_top_est[i])  for i in range(4)]
            self.top_to_bottom_disparity = [self.generate_image_bottom(tf.abs(self.disparity_top_est[i]), self.disparity_bottom_est[i]) for i in range(4)]

        # Edge-aware smoothness.
        with tf.variable_scope('smoothness'):
            if self.params.dual_loss:
                self.depth_top_smoothness = self.get_smoothness([tf.log(1.0 + tf.abs(depth)) for depth in self.depth_top_est], self.top_pyramid)
                self.depth_bottom_smoothness = self.get_smoothness([tf.log(1.0 + tf.abs(depth)) for depth in self.depth_bottom_est], self.bottom_pyramid)

            self.disparity_top_smoothness  = self.get_smoothness(self.disparity_top_est,  self.top_pyramid)
            self.disparity_bottom_smoothness = self.get_smoothness(self.disparity_bottom_est, self.bottom_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse = self.reuse_variables):
            # L1
            self.l1_top = [tf.abs(self.top_est[i] - self.top_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_top  = [tf.reduce_mean(l) for l in self.l1_top]
            self.l1_bottom = [tf.abs(self.bottom_est[i] - self.bottom_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_bottom = [tf.reduce_mean(l) for l in self.l1_bottom]

            # SSIM
            self.ssim_top = [self.SSIM(self.top_est[i],  self.top_pyramid[i]) for i in range(4)]
            self.ssim_loss_top  = [tf.reduce_mean(s) for s in self.ssim_top]
            self.ssim_bottom = [self.SSIM(self.bottom_est[i], self.bottom_pyramid[i]) for i in range(4)]
            self.ssim_loss_bottom = [tf.reduce_mean(s) for s in self.ssim_bottom]

            # WEIGTHED SUM
            self.image_loss_bottom = [self.params.alpha_image_loss * self.ssim_loss_bottom[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_bottom[i] for i in range(4)]
            self.image_loss_top  = [self.params.alpha_image_loss * self.ssim_loss_top[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_top[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_top + self.image_loss_bottom)

            # DISPARITY SMOOTHNESS
            self.disparity_top_loss  = [tf.reduce_mean(tf.abs(self.disparity_top_smoothness[i]))  / 2 ** i for i in range(4)]
            self.disparity_bottom_loss = [tf.reduce_mean(tf.abs(self.disparity_bottom_smoothness[i])) / 2 ** i for i in range(4)]
            self.disparity_gradient_loss = tf.add_n(self.disparity_top_loss + self.disparity_bottom_loss)
            
            if self.params.dual_loss:
                self.depth_top_loss  = [tf.reduce_mean(tf.abs(self.depth_top_smoothness[i]))  / 2 ** i for i in range(4)]
                self.depth_bottom_loss  = [tf.reduce_mean(tf.abs(self.depth_bottom_smoothness[i]))  / 2 ** i for i in range(4)]
                self.depth_gradient_loss = tf.add_n(self.depth_top_loss + self.depth_bottom_loss)
                self.smoothness_loss = 0.25 * self.depth_gradient_loss + self.disparity_gradient_loss
            else:
                self.smoothness_loss = self.disparity_gradient_loss
            
            # TB CONSISTENCY
            self.tb_top_loss  = [tf.reduce_mean(tf.abs(self.bottom_to_top_disparity[i] - tf.abs(self.disparity_top_est[i])))  for i in range(4)]
            self.tb_bottom_loss = [tf.reduce_mean(tf.abs(self.top_to_bottom_disparity[i] - tf.abs(self.disparity_bottom_est[i]))) for i in range(4)]
            if self.params.dual_loss:
                self.tb_top_loss_depth  = [0.25 * tf.reduce_mean(tf.abs(self.bottom_to_top_depth[i] - tf.log(1.0 + tf.abs(self.depth_top_est[i]))))  for i in range(4)]
                self.tb_bottom_loss_depth = [0.25 * tf.reduce_mean(tf.abs(self.top_to_bottom_depth[i] - tf.log(1.0 + tf.abs(self.depth_bottom_est[i])))) for i in range(4)]
                self.tb_loss = tf.add_n(self.tb_top_loss + self.tb_bottom_loss + self.tb_top_loss_depth + self.tb_bottom_loss_depth)
            else:
                self.tb_loss = tf.add_n(self.tb_top_loss + self.tb_bottom_loss)

            self.depth_metrics = self.get_metrics(self.depth_top_est[0])
            self.disparity_metrics = self.get_metrics(self.disparity_top_est[0])

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.params.smoothness_loss_weight * self.smoothness_loss + self.params.tb_loss_weight * self.tb_loss

    # Normalize images to be between 0 and 1.
    def normalize_image(self, input_images):
        max = tf.reduce_max(input_images, axis = [1, 2], keep_dims = True)
        min = tf.reduce_min(input_images, axis = [1, 2], keep_dims = True)
        return (input_images - min) / (max - min)

    def get_metrics(self, input_images):
        min = tf.reduce_min(input_images)
        max = tf.reduce_max(input_images)
        mean = tf.reduce_mean(input_images)
        return min, max, mean

    def build_summaries(self):
        with tf.device('/cpu:0'):
            # Scalar summaries.
            tf.summary.scalar('ssim_loss', self.ssim_loss_top[0] + self.ssim_loss_bottom[0], collections=self.model_collection)
            tf.summary.scalar('l1_loss', self.l1_reconstruction_loss_top[0] + self.l1_reconstruction_loss_bottom[0], collections=self.model_collection)
            tf.summary.scalar('image_loss', self.image_loss_top[0] + self.image_loss_bottom[0], collections=self.model_collection)
            tf.summary.scalar('smoothness_loss', self.disparity_top_loss[0] + self.disparity_bottom_loss[0], collections=self.model_collection)
            tf.summary.scalar('tb_loss', self.tb_top_loss[0] + self.tb_bottom_loss[0], collections=self.model_collection)

            # Depth/disparity ranges.
            tf.summary.scalar('depth_min', tf.reshape(self.depth_metrics[0], []), collections = self.model_collection)
            tf.summary.scalar('depth_max', tf.reshape(self.depth_metrics[1], []), collections = self.model_collection)
            tf.summary.scalar('depth_mean', tf.reshape(self.depth_metrics[2], []), collections = self.model_collection)
            tf.summary.scalar('disparity_min', tf.reshape(self.disparity_metrics[0], []), collections = self.model_collection)
            tf.summary.scalar('disparity_max', tf.reshape(self.disparity_metrics[1], []), collections = self.model_collection)
            tf.summary.scalar('disparity_mean', tf.reshape(self.disparity_metrics[2], []), collections = self.model_collection)

            # Network outputs.
            tf.summary.image('disparity_top_est', tf.abs(self.disparity_top_est[0]), max_outputs=4, collections = self.model_collection)
            tf.summary.image('disparity_bottom_est', tf.abs(self.disparity_bottom_est[0]), max_outputs=4, collections = self.model_collection)
            tf.summary.image('depth_top_est', normalize_depth(perpendicular_to_distance(self.depth_top_est[0])), max_outputs=4, collections = self.model_collection)
            tf.summary.image('depth_bottom_est', normalize_depth(perpendicular_to_distance(self.depth_bottom_est[0])), max_outputs = 4, collections = self.model_collection)

            # Image reconstruction summaries.
            tf.summary.image('top_est', self.top_est[0], max_outputs = 4, collections = self.model_collection)
            tf.summary.image('bottom_est', self.bottom_est[0], max_outputs = 4, collections = self.model_collection)
            tf.summary.image('ssim_top', self.ssim_top[0],  max_outputs = 4, collections = self.model_collection)
            tf.summary.image('ssim_bottom', self.ssim_bottom[0], max_outputs = 4, collections = self.model_collection)
            tf.summary.image('l1_top', self.l1_top[0],  max_outputs = 4, collections = self.model_collection)
            tf.summary.image('l1_bottom', self.l1_bottom[0], max_outputs = 4, collections = self.model_collection)
            tf.summary.image('top',  self.top_pyramid[0],   max_outputs = 4, collections = self.model_collection)
            tf.summary.image('bottom', self.bottom_pyramid[0],  max_outputs = 4, collections = self.model_collection)
