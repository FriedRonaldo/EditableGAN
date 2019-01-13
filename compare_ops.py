# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
from six.moves import map
from six.moves import range
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import add_arg_scope

import math
from scipy import signal


def check_folder(log_dir):
    if not tf.gfile.IsDirectory(log_dir):
        tf.gfile.MakeDirs(log_dir)
    return log_dir


def save_images(images, image_path):
    with tf.gfile.Open(image_path, "wb") as f:
        scipy.misc.imsave(f, images * 255.0)


def lrelu(input_, leak=0.2, name="lrelu"):
    return tf.maximum(input_, leak * input_, name=name)


def batch_norm(input_, is_training, scope):
    return tf.contrib.layers.batch_norm(
        input_,
        decay=0.999,
        epsilon=0.001,
        updates_collections=None,
        scale=True,
        fused=False,
        is_training=is_training,
        scope=scope)


def layer_norm(input_, scope):
    return tf.contrib.layers.layer_norm(input_, trainable=True, scope=scope)


def instance_norm(input_, scope):
    return tf.contrib.layers.instance_norm(input_, trainable=True, scope=scope)


def pixel_norm(input_, scope):
    with tf.variable_scope(scope):
        if len(input_.shape) > 2:
            _axis = 3
        else:
            _axis = 1
        return input_ * tf.rsqrt(tf.reduce_mean(tf.square(input_), axis=_axis, keepdims=True) + 0.0001)


def group_norm(input_, scope):
    return tf.contrib.layers.group_norm(input_, epsilon=0.0003, scope=scope)


# def group_norm(x, name='group_norm'):
#     with tf.variable_scope(name):
#         G = 32
#         eps = 1e-5
#         N, H, W, C = x.get_shape().as_list()
#         G = min(G, C)
#
#         x = tf.reshape(x, [N, H, W, G, C // G])
#         mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
#         x = (x - mean) / tf.sqrt(var + eps)
#
#         gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
#         beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))
#
#         x = tf.reshape(x, [N, H, W, C]) * gamma + beta
#
#     return x


def norms(input_, is_train, scope, norm_type=None):
    assert norm_type in ['batch', 'layer', 'instance', 'pixel', 'group', None]
    if norm_type == 'batch':
        return batch_norm(input_, is_train, scope)
    elif norm_type == 'layer':
        return layer_norm(input_, scope)
    elif norm_type == 'instance':
        return instance_norm(input_, scope)
    elif norm_type == 'pixel':
        return pixel_norm(input_, scope)
    elif norm_type == 'group':
        return group_norm(input_, scope)
    else:
        return input_


def acts(input_, act_type=None):
    assert act_type in ['relu', 'lrelu', 'tanh', 'sigmoid', None]
    if act_type == 'relu':
        return tf.nn.relu(input_)
    elif act_type == 'lrelu':
        return tf.nn.leaky_relu(input_, 0.1)
    elif act_type == 'tanh':
        return tf.nn.tanh(input_)
    elif act_type =='sigmoid':
        return tf.nn.sigmoid(input_)
    else:
        return input_


def spectral_norm(input_):
    """Performs Spectral Normalization on a weight tensor."""
    if len(input_.shape) < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf.reshape(input_, (-1, input_.shape[-1]))

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = tf.get_variable(
        input_.name.replace(":", "") + "/u_var",
        shape=(w.shape[0], 1),
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.nn.l2_normalize(
            tf.matmul(tf.transpose(w), u), axis=None, epsilon=1e-12)
        u = tf.nn.l2_normalize(tf.matmul(w, v), axis=None, epsilon=1e-12)

    # Update persisted approximation.
    with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, input_.shape)
    return w_tensor_normalized


def spectral_norm_update_ops(var_list, weight_getter):
    update_ops = []
    print(" [*] Spectral norm layers")
    layer = 0
    for var in var_list:
        if weight_getter.match(var.name):
            layer += 1
            print("     %d. %s" % (layer, var.name))
            # Alternative solution here is keep spectral norm and original weight
            # matrix separately, and only normalize the weight matrix if needed.
            # But as spectral norm converges to 1.0 very quickly, it should be very
            # minor accuracy diff caused by float value division.
            update_ops.append(tf.assign(var, spectral_norm(var)))
    return update_ops


def spectral_norm_svd(input_):
    if len(input_.shape) < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors")

    w = tf.reshape(input_, (-1, input_.shape[-1]))
    s, _, _ = tf.svd(w)
    return s[0]


def spectral_norm_value(var_list, weight_getter):
    """Compute spectral norm value using svd, for debug purpose."""
    norms = {}
    for var in var_list:
        if weight_getter.match(var.name):
            norms[var.name] = spectral_norm_svd(var)
    return norms


def linear(input_,
           output_size,
           scope=None,
           stddev=0.02,
           bias_start=0.0,
           use_sn=False):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(
            "bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if use_sn:
            return tf.matmul(input_, spectral_norm(matrix)) + bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           initializer=tf.truncated_normal_initializer, use_sn=False, xavier=False):
    with tf.variable_scope(name):
        if xavier:
            w = tf.get_variable(
                "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True), )
        else:
            w = tf.get_variable(
                "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                initializer=initializer(stddev=stddev))
        if use_sn:
            conv = tf.nn.conv2d(
                input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
        biases = tf.get_variable(
            "biases", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w,
             stddev=0.02, name="deconv2d", use_sn=False, xavier=False):
    with tf.variable_scope(name):
        if xavier:
            w = tf.get_variable(
                "w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        else:
            w = tf.get_variable(
                "w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.random_normal_initializer(stddev=stddev))
        if use_sn:
            deconv = tf.nn.conv2d_transpose(
                input_, spectral_norm(w), output_shape=output_shape, strides=[1, d_h, d_w, 1])
        else:
            deconv = tf.nn.conv2d_transpose(
                input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable(
            "biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


def weight_norm_linear(input_, output_size,
                       init=False, init_scale=1.0,
                       name="wn_linear",
                       initializer=tf.truncated_normal_initializer,
                       stddev=0.02):
    """Linear layer with Weight Normalization (Salimans, Kingma '16)."""
    with tf.variable_scope(name):
        if init:
            v = tf.get_variable("V", [int(input_.get_shape()[1]), output_size],
                                tf.float32, initializer(0, stddev), trainable=True)
            v_norm = tf.nn.l2_normalize(v.initialized_value(), [0])
            x_init = tf.matmul(input_, v_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable("g", dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable("b", dtype=tf.float32, initializer=
            -m_init*scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [1, output_size]) * (
                    x_init - tf.reshape(m_init, [1, output_size]))
            return x_init
        else:

            v = tf.get_variable("V")
            g = tf.get_variable("g")
            b = tf.get_variable("b")
            tf.assert_variables_initialized([v, g, b])
            x = tf.matmul(input_, v)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(v), [0]))
            x = tf.reshape(scaler, [1, output_size]) * x + tf.reshape(
                b, [1, output_size])
            return x


def weight_norm_conv2d(input_, output_dim,
                       k_h, k_w, d_h, d_w,
                       init, init_scale,
                       stddev=0.02,
                       name="wn_conv2d",
                       initializer=tf.truncated_normal_initializer):
    """Convolution with Weight Normalization (Salimans, Kingma '16)."""
    with tf.variable_scope(name):
        if init:
            v = tf.get_variable(
                "V", [k_h, k_w] + [int(input_.get_shape()[-1]), output_dim],
                tf.float32, initializer(0, stddev), trainable=True)
            v_norm = tf.nn.l2_normalize(v.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(input_, v_norm, strides=[1, d_h, d_w, 1],
                                  padding="SAME")
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable(
                "g", dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable(
                "b", dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, 1, output_dim]) * (
                    x_init - tf.reshape(m_init, [1, 1, 1, output_dim]))
            return x_init
        else:
            v = tf.get_variable("V")
            g = tf.get_variable("g")
            b = tf.get_variable("b")
            tf.assert_variables_initialized([v, g, b])
            w = tf.reshape(g, [1, 1, 1, output_dim]) * tf.nn.l2_normalize(
                v, [0, 1, 2])
            x = tf.nn.bias_add(
                tf.nn.conv2d(input_, w, [1, d_h, d_w, 1], padding="SAME"), b)
            return x


def weight_norm_deconv2d(x, output_dim,
                         k_h, k_w, d_h, d_w,
                         init=False, init_scale=1.0,
                         stddev=0.02,
                         name="wn_deconv2d",
                         initializer=tf.truncated_normal_initializer):
    """Transposed Convolution with Weight Normalization (Salimans, Kingma '16)."""
    xs = list(map(int, x.get_shape()))
    target_shape = [xs[0], xs[1] * d_h, xs[2] * d_w, output_dim]
    with tf.variable_scope(name):
        if init:
            v = tf.get_variable(
                "V", [k_h, k_w] + [output_dim, int(x.get_shape()[-1])],
                tf.float32, initializer(0, stddev), trainable=True)
            v_norm = tf.nn.l2_normalize(v.initialized_value(), [0, 1, 3])
            x_init = tf.nn.conv2d_transpose(x, v_norm, target_shape,
                                            [1, d_h, d_w, 1], padding="SAME")
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale/tf.sqrt(v_init + 1e-8)
            g = tf.get_variable("g", dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable("b", dtype=tf.float32,
                                initializer=-m_init*scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, 1, output_dim]) * (
                    x_init - tf.reshape(m_init, [1, 1, 1, output_dim]))
            return x_init
        else:
            v = tf.get_variable("v")
            g = tf.get_variable("g")
            b = tf.get_variable("b")
            tf.assert_variables_initialized([v, g, b])
            w = tf.reshape(g, [1, 1, output_dim, 1]) * tf.nn.l2_normalize(
                v, [0, 1, 3])
            x = tf.nn.conv2d_transpose(x, w, target_shape, strides=[1, d_h, d_w, 1],
                                       padding="SAME")
            x = tf.nn.bias_add(x, b)
            return x


# From https://github.com/tensorflow/tensorflow/issues/2169
def unpool(value, name="unpool"):
    """Unpooling operation.

    N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    Taken from: https://github.com/tensorflow/tensorflow/issues/2169

    Args:
      value: a Tensor of shape [b, d0, d1, ..., dn, ch]
      name: name of the op

    Returns:
      A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        value = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            value = tf.concat([value, tf.zeros_like(value)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        value = tf.reshape(value, out_size, name=scope)
    return value


@add_arg_scope
def unpool_2d(pool,
              ind,
              stride=[1, 2, 2, 1],
              scope='unpool_2d'):
  """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """
  with tf.variable_scope(scope):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                      shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret


def pool(input_, name=None):
    output = tf.nn.pool(input_, [2, 2], "AVG", "SAME", strides=[2, 2], name=name)
    return output


def upsample(input_):
    size = input_.get_shape().as_list()
    return tf.image.resize_images(input_, [size[1] * 2, size[2] * 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)


# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
def minibatch_stddev_block(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, x.shape[0].value)       # Minibatch must be divisible by (or smaller than) group_size.
        sh = x.shape                                                # [NHWC]    Input shape
        y = tf.reshape(x, [group_size, -1, sh[1], sh[2], sh[3]])    # [GMHWC]   Split batch into M groups of size G
        y = tf.cast(y, tf.float32)                                  # [GMHWC]   Cast to float32
        y -= tf.reduce_mean(y, axis=0, keepdims=True)               # [GMHWC]   Subtract mean over group
        y = tf.reduce_mean(tf.square(y), axis=0)                    # [MHWC]    Calc variance over group
        y = tf.sqrt(y + 1e-8)                                       # [MHWC]    Calc stddev over group
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)        # [M111]    Take avg. over feature maps and pix
        y = tf.cast(y, x.dtype)                                     # [M111]    Cast to the type of x
        y = tf.tile(y, [group_size, sh[1], sh[2], 1])               # [NHW1]    Replicate over group and pixels
        return tf.concat([x, y], axis=3)                            # [NHWC']   Append the new feature map


def weighted_sigmoid_ce_with_logits(targets, logits, pos_weight, neg_weight, name=None):
    # loss = targets * -tf.log(tf.nn.sigmoid(logits)) * pos_weight + (1-targets) * -tf.log(1 - tf.nn.sigmoid(logits)) * neg_weight
    with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
        logits = ops.convert_to_tensor(logits, name='logits')
        targets = ops.convert_to_tensor(targets, name='targets')
        log_weight = (neg_weight + targets * (pos_weight - neg_weight))
        # loss = neg_weight * (1 - targets) * logits + (targets * (pos_weight - neg_weight) + neg_weight) * tf.log(1 + tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)
        return math_ops.add(
            neg_weight * (1 - targets) * logits,
            log_weight * (math_ops.log1p(1 + math_ops.exp(-math_ops.abs(logits))) +
                          nn_ops.relu(-logits)), name=name)


if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def eval_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def eval_ssim(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small
      images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).

    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def compute_mean_covariance(img):
    shape = img.get_shape().as_list()

    numBatchs = shape[0]

    numPixels = shape[1] * shape[2]

    cDims = shape[3]

    mu = tf.reduce_mean(img, axis=[1, 2])

    img_hat = img - tf.reshape(mu, shape=[mu.shape[0], 1, 1, mu.shape[1]])

    img_hat = tf.reshape(img_hat, [numBatchs, cDims, numPixels])

    cov = tf.matmul(img_hat, img_hat, transpose_b=True)

    cov = cov / numPixels

    return mu, cov


def generate_y_rand(size):
    # bang  black   blond   brown   gray    male    mustache     smile   glasses young
    # 12/60 15/60   10/60   12/60   6/60    30/60   6/60         30/60   6/60    48/60
    total_size = size
    tmp_rand = []
    tmp_rand.append(np.random.choice(2, total_size, p=[48. / 60., 12. / 60.]))
    tmp = np.random.choice(5, total_size, p=[17. / 60., 15. / 60., 10. / 60., 12. / 60., 6. / 60.])
    tmp_z = np.zeros([total_size, 4])
    i = 0
    for m in tmp:
        if m != 0:
            tmp_z[i][m - 1] = 1
        i += 1

    tmp_z = tmp_z.transpose()
    for z in tmp_z:
        tmp_rand.append(z)
    tmp_rand.append(np.random.choice(2, total_size, p=[30. / 60., 30. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[54. / 60., 6. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[30. / 60., 30. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[54. / 60., 6. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[12. / 60., 48. / 60.]))

    y_rand_total = np.asarray(tmp_rand).transpose()
    return y_rand_total.astype(np.float32)