import tensorflow as tf
from utils import logger
import compare_ops as ops


class SNGenerator(object):
    def __init__(self, name, norm='batch', activation='relu',
                 image_shape=[128, 128, 3], batch_size=64, sn=False):
        logger.info('Init Generator %s', name)
        self.name = name
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._image_shape = image_shape
        self._batch_size = batch_size
        self._sn = sn
        self._nf = 64

    def __call__(self, z, y=None, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            if y is not None:
                z = tf.concat([z, y], 1)
                y = tf.reshape(y, [self._batch_size, 1, 1, y.shape[-1].value])
            # 16*16*512
            G = ops.linear(z, 16*16*self._nf*8, 'linear1', use_sn=self._sn)
            # 16*16*512
            G = tf.reshape(G, [-1, 16, 16, self._nf*8])
            G = ops.norms(G, is_train, 'norm1', self._norm)
            G = ops.acts(G, self._activation)
            # 32 32 256
            if y is not None:
                G = ops.conv_cond_concat(G, y)
            G = self.block('block1', G, self._nf*4, 32, is_train)
            # 64 64 128
            if y is not None:
                G = ops.conv_cond_concat(G, y)
            G = self.block('block2', G, self._nf*2, 64, is_train)
            # 128 128 64
            if y is not None:
                G = ops.conv_cond_concat(G, y)
            G = self.block('block3', G, self._nf, 128, is_train)
            # 128 128 3
            G = self.block('block4', G, 3, 128, is_train, True)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            return G

    def block(self, name, x, chout, spout, is_train, is_last=False):
        with tf.variable_scope(name, reuse=self._reuse):
            if not is_last:
                G = ops.deconv2d(x, [self._batch_size, spout, spout, chout], 4, 4, 2, 2, name='conv', use_sn=self._sn, xavier=True)
                G = ops.norms(G, is_train, 'norm', self._norm)
                G = ops.acts(G, self._activation)
            else:
                G = ops.deconv2d(x, [self._batch_size, spout, spout, chout], 3, 3, 1, 1, name='conv', use_sn=self._sn, xavier=True)
                G = ops.acts(G, 'tanh')

            return G


class MNISTGenerator(object):
    def __init__(self, name, norm='batch', activation='relu',
                 image_shape=[28, 28, 1], batch_size=128, sn=False):
        logger.info('Init MNISTGenerator %s', name)
        self.name = name
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._image_shape = image_shape
        self._batch_size = batch_size
        self._sn = sn
        self._nf = 128

    def __call__(self, z, y=None, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            if y is not None:
                z = tf.concat([z, y], 1)
            # 4*4*512
            G = ops.linear(z, 4 * 4 * self._nf * 4, 'linear')
            # 4 4 512
            G = tf.reshape(G, [-1, 4, 4, self._nf * 4])
            G = ops.norms(G, is_train, 'norm', self._norm)
            G = ops.acts(G, self._activation)
            # 7 7 256
            G = self.block('block1', G, self._nf * 2, 7, is_train)
            # 14 14 128
            G = self.block('block2', G, self._nf, 14, is_train)
            # 28 28 1
            G = self.block('block3', G, 1, 28, is_train, True)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            return G

    def block(self, name, x, chout, spout, is_train, is_last=False):
        with tf.variable_scope(name, reuse=self._reuse):
            G = ops.deconv2d(x, [self._batch_size, spout, spout, chout], 5, 5, 2, 2, name='conv', use_sn=self._sn, xavier=True)
            if not is_last:
                G = ops.norms(G, is_train, 'norm', self._norm)
                G = ops.acts(G, self._activation)
            else:
                G = ops.acts(G, 'tanh')

            return G

