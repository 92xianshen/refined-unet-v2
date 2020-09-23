"""
MIT License

Copyright (c) 2020 Libin Jiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf


def _diagonal_compatibility(shape):
    return tf.eye(shape[0], shape[1], dtype=np.float32)


def _potts_compatibility(shape):
    return -1 * _diagonal_compatibility(shape)


class GaussianLayer(tf.keras.layers.Layer):
    """ Gaussian layer """

    def __init__(self, theta_gamma):
        super(GaussianLayer, self).__init__()
        self.theta_gamma = theta_gamma
        self.radius = int(theta_gamma * 4)

        x, y = np.mgrid[-self.radius:self.radius +
                        1, -self.radius:self.radius + 1]
        k = np.exp(-(x ** 2 + y ** 2) / (2 * theta_gamma ** 2))
        k = k / k.sum()
        k[self.radius, self.radius] = 0
        self.kernel = tf.constant(k, dtype=tf.float32)

    def call(self, src):
        batch_size, height, width, channels = src.shape

        kernel = tf.repeat(
            self.kernel[..., tf.newaxis, tf.newaxis], repeats=channels, axis=2)
        src = tf.pad(src, [[0, 0], [self.radius, self.radius], [
                     self.radius, self.radius], [0, 0]], mode='constant', constant_values=0)

        dst = tf.nn.depthwise_conv2d(
            src, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return dst


class GuidedGaussianLayer(tf.keras.layers.Layer):
    """ Guided filter layer """

    def __init__(self, radius, eps):
        super(GuidedGaussianLayer, self).__init__()
        self.radius = radius
        self.eps = eps
        self.theta_alpha = radius / 3.0
        x, y = np.mgrid[-self.radius:self.radius +
                        1, -self.radius:self.radius + 1]
        k = np.exp(-(x ** 2 + y ** 2) / (2 * self.theta_alpha ** 2))
        k = k / k.sum()
        k[self.radius, self.radius] = 0
        self.kernel = tf.constant(k, dtype=np.float32)

    def call(self, I, p):
        def gaussian_filter(src, r):
            channels = tf.shape(src)[-1]
            kernel = tf.repeat(
                self.kernel[..., tf.newaxis, tf.newaxis], repeats=channels, axis=2)
            src = tf.pad(src, [[0, 0], [r, r], [r, r], [0, 0]],
                         mode='constant', constant_values=0)
            dst = tf.nn.depthwise_conv2d(
                src, kernel, strides=[1, 1, 1, 1], padding='VALID')

            return dst

        def guided_filter(I, p, r, eps):
            mean_I = gaussian_filter(I, r)
            mean_p = gaussian_filter(p, r)
            mean_Ip = gaussian_filter(I * p, r)
            cov_Ip = mean_Ip - mean_I * mean_p

            mean_II = gaussian_filter(I * I, r)
            var_I = mean_II - mean_I * mean_I

            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            mean_a = gaussian_filter(a, r)
            mean_b = gaussian_filter(b, r)

            q = mean_a * I + mean_b

            return q

        assert len(I.shape) == 4 and len(p.shape) == 4
        gray = tf.image.rgb_to_grayscale(I)
        q = guided_filter(gray, p, self.radius, self.eps)
        return q


class CRFLayer(tf.keras.layers.Layer):
    """ A layer implementing Dense CRF """

    def __init__(self, num_classes, radius, eps, theta_gamma, spatial_compat, bilateral_compat, num_iterations):
        super(CRFLayer, self).__init__()
        self.num_classes = num_classes
        self.radius = radius
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations

        self.spatial_weights = spatial_compat * \
            _diagonal_compatibility((num_classes, num_classes))
        self.bilateral_weights = bilateral_compat * \
            _diagonal_compatibility((num_classes, num_classes))
        self.compatibility_matrix = _potts_compatibility(
            (num_classes, num_classes))

        self.bilateral = GuidedGaussianLayer(radius, eps)
        self.gaussian = GaussianLayer(theta_gamma)

    def call(self, unary, image):
        assert len(image.shape) == 4 and len(unary.shape) == 4

        unary_shape = tf.shape(unary)
        all_ones = tf.ones((unary_shape[0], unary_shape[1], unary_shape[2], self.num_classes), dtype=tf.float32)
        spatial_norm_vals = self.gaussian(all_ones)
        bilateral_norm_vals = self.bilateral(image, all_ones)

        # Initialize Q
        Q = tf.nn.softmax(-unary)

        for i in range(self.num_iterations):
            tmp1 = -unary

            # Message passing - spatial
            spatial_out = self.gaussian(Q)
            spatial_out /= spatial_norm_vals

            # Message passing - bilateral
            bilateral_out = self.bilateral(image, Q)
            bilateral_out /= bilateral_norm_vals

            # Message passing
            spatial_out = tf.reshape(spatial_out, [-1, self.num_classes])
            spatial_out = tf.matmul(spatial_out, self.spatial_weights)
            bilateral_out = tf.reshape(bilateral_out, [-1, self.num_classes])
            bilateral_out = tf.matmul(bilateral_out, self.bilateral_weights)
            message_passing = spatial_out + bilateral_out

            # Compatibility transform
            pairwise = tf.matmul(message_passing, self.compatibility_matrix)
            pairwise = tf.reshape(
                pairwise, [unary_shape[0], unary_shape[1], unary_shape[2], self.num_classes])

            # Local update
            tmp1 -= pairwise

            # Normalize
            Q = tf.nn.softmax(tmp1)

        return Q
