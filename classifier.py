import tensorflow as tf
from utils import logger
import compare_ops as ops


class V3Classifier(object):
    def __init__(self, name, norm='batch', activation='lrelu', batch_size=64, num_labels=10, sn=False):
        logger.info('Init Classifier %s', name)
        self.name = name
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._batch_size = batch_size
        self._sn = sn
        self._num_labels = num_labels
        self.ch_base = 32

    def __call__(self, x, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            # 128 128 64 -> 64 64 128
            C = self.block('block1', x, self.ch_base, is_train, is_first=True)
            # 64 64 128 -> 32 32 256
            C = self.block('block2', C, self.ch_base*2, is_train)
            # 32 32 256-> 16 16 512
            C = self.block('block3', C, self.ch_base*4, is_train)
            # 16 16 512
            C = Cn = self.block('block4', C, self.ch_base*8, is_train, is_last=True)
            # 512 : GAP
            C = tf.reduce_mean(C, axis=[1, 2])
            # 256
            C = self.FCblock('block7', C, self.ch_base*8, is_train)
            # 11
            C = self.FCblock('block8', C, self._num_labels + 1, is_train, True)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            return C[:, :self._num_labels], Cn, C[:, self._num_labels:]

    def FCblock(self, name, x, chout, is_train, is_last=False):
        with tf.variable_scope(name, reuse=self._reuse):
            C = ops.linear(x, chout, 'linear')
            if not is_last:
                C = ops.norms(C, is_train, 'norm', self._norm)
                C = ops.acts(C, self._activation)
            return C

    def block(self, name, x, chout, is_train, is_last=False, is_first=False):
        with tf.variable_scope(name, reuse=self._reuse):
            D = ops.conv2d(x, chout, 3, 3, 1, 1, 0.02, 'conv1', use_sn=self._sn, xavier=True)
            if not is_first:
                D = ops.norms(D, is_train, 'norm1', self._norm)
            D = ops.acts(D, self._activation)

            if not is_last:
                D = ops.conv2d(D, chout*2, 4, 4, 2, 2, 0.02, 'conv2', use_sn=self._sn, xavier=True)
                if not is_first:
                    D = ops.norms(D, is_train, 'norm2', self._norm)
                D = ops.acts(D, self._activation)

            return D


class MNISTClassifier(object):
    def __init__(self, name, norm='batch', activation='lrelu', batch_size=128, num_labels=10, sn=False):
        logger.info('Init Classifier %s', name)
        self.name = name
        self._norm = norm
        self._activation = activation
        self._reuse = False
        self._batch_size = batch_size
        self._sn = sn
        self._num_labels = num_labels
        self._nf = 128

    def __call__(self, x, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            # 14 14 128
            C = self.block('block1', x, self._nf, is_train, False)
            # 7 7 256
            C = self.block('block2', C, self._nf * 2, is_train)
            # 4 4 512
            C = Cn = self.block('block3', C, self._nf * 4, is_train)
            # 4*4*512
            C = tf.reshape(C, [self._batch_size, -1])
            # 10
            C = ops.linear(C, self._num_labels, 'linear')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            return C, Cn

    def block(self, name, x, chout, is_train, use_norm=True):
        with tf.variable_scope(name, reuse=self._reuse):
            C = ops.conv2d(x, chout, 5, 5, 2, 2, 0.02, 'conv', use_sn=self._sn, xavier=True)
            if use_norm:
                C = ops.norms(C, is_train, 'norm', self._norm)
            C = ops.acts(C, self._activation)

            return C
