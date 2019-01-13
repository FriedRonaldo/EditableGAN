import tensorflow as tf
from utils import logger
import compare_ops as ops


class SNDiscriminator(object):
    def __init__(self, name, norm=None, activation='lrelu', batch_size=64, sn=False):
        logger.info('Init SNDiscriminator %s', name)
        self.name = name
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._batch_size = batch_size
        self._sn = sn
        self._nf = 64

    def __call__(self, x, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            # 128 128 64 -> 64 64 128
            D = self.block('block1', x, self._nf)
            # 64 64 128 -> 32 32 256
            D = self.block('block2', D, self._nf*2)
            # 32 32 256 -> 16 16 512
            D = self.block('block3', D, self._nf*4)
            # 16 16 512
            D = Cn = self.block('block4', D, self._nf*8, True)
            # 512 / GAP
            D = tf.reduce_mean(D, axis=[1, 2])
            # 1
            D = ops.linear(D, 1, 'linear', use_sn=self._sn)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            return D, Cn

    def block(self, name, x, chout, is_last=False):
        with tf.variable_scope(name, reuse=self._reuse):
            D = ops.conv2d(x, chout, 3, 3, 1, 1, 0.02, 'conv1', use_sn=self._sn, xavier=True)
            D = ops.acts(D, self._activation)
            if not is_last:
                D = ops.conv2d(D, chout*2, 4, 4, 2, 2, 0.02, 'conv2', use_sn=self._sn, xavier=True)
                D = ops.acts(D, self._activation)

            return D


class MNISTDiscriminator(object):
    def __init__(self, name, norm='batch', activation='lrelu', batch_size=128, sn=False):
        logger.info('Init Discriminator %s', name)
        self.name = name
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._batch_size = batch_size
        self._sn = sn
        self._nf = 128

    def __call__(self, x, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            # 14 14 128
            D = self.block('block1', x, self._nf, is_train, False)
            # 7 7 256
            D = self.block('block2', D, self._nf * 2, is_train)
            # 4 4 512
            D = Cn = self.block('block3', D, self._nf * 4, is_train)
            # 4*4*512
            D = tf.reshape(D, [self._batch_size, -1])
            # 1
            D = ops.linear(D, 1, 'linear')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            return D, Cn

    def block(self, name, x, chout, is_train, use_norm=True):
        with tf.variable_scope(name, reuse=self._reuse):
            D = ops.conv2d(x, chout, 5, 5, 2, 2, 0.02, 'conv1', use_sn=self._sn, xavier=True)
            if use_norm:
                D = ops.norms(D, is_train, 'norm', self._norm)
            D = ops.acts(D, self._activation)

            return D


