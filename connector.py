import tensorflow as tf
from utils import logger
import compare_ops as ops


class V3CondConnector(object):
    def __init__(self, name, norm=None, activation='lrelu', batch_size=64, latent_size=128, num_labels=10):
        logger.info('Init Connector %s', name)
        self.name = name
        self._batch_size = batch_size
        self._latent_size = latent_size
        self._num_labels = num_labels
        self._norm = norm
        self._activation = activation
        self._reuse = False

    def __call__(self, x_d, x_c, y=None, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            # x_d : 16 16 512
            # x_c : 16 16 512
            if y is not None:
                y = tf.reshape(y, [self._batch_size, 1, 1, y.shape[-1].value])
                x_d = ops.conv_cond_concat(x_d, y)
                x_c = ops.conv_cond_concat(x_c, y)
                x = tf.concat([x_d, x_c], axis=3)
            else:
                x = tf.concat([x_d, x_c], axis=3)
            # x : 16 16 1024 + alpha
            # 8 8 1024
            Cn = self.convblock('block1', x, 1024, is_train)
            # 4 4 1024
            Cn = self.convblock('block2', Cn, 1024, is_train)
            # 2 2 1024
            Cn = self.convblock('block3', Cn, 1024, is_train)
            # 1024 : GSP
            Cn = tf.reduce_sum(Cn, axis=[1, 2])
            Cn = self.block('block4', Cn, self._latent_size + self._num_labels, is_train, True)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            return Cn[:, :self._latent_size], Cn[:, self._latent_size:]

    def block(self, name, x, chout, is_train, is_last=False):
        with tf.variable_scope(name, reuse=self._reuse):
            Cn = ops.linear(x, chout, 'linear')
            if not is_last:
                Cn = ops.norms(Cn, is_train, 'norm', self._norm)
                Cn = ops.acts(Cn, self._activation)
            return Cn

    def convblock(self, name, x, chout, is_train, is_last=False, s=2):
        with tf.variable_scope(name, reuse=self._reuse):
            Cn = ops.conv2d(x, chout, 4, 4, s, s, xavier=True)
            if not is_last:
                Cn = ops.norms(Cn, is_train, 'norm', self._norm)
                Cn = ops.acts(Cn, self._activation)
            return Cn


class MNISTConnector(object):
    def __init__(self, name, norm='batch', activation='lrelu', batch_size=64, latent_size=128, num_labels=10):
        logger.info('Init Connector %s', name)
        self.name = name
        self._batch_size = batch_size
        self._latent_size = latent_size
        self._num_labels = num_labels
        self._norm = norm
        self._activation = activation
        self._reuse = False

    def __call__(self, x_d, x_c, y=None, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            # x_d : 512
            # x_c : 512
            if y is not None:
                y = tf.reshape(y, [self._batch_size, 1, 1, y.shape[-1].value])
                x_d = ops.conv_cond_concat(x_d, y)
                x_c = ops.conv_cond_concat(x_c, y)
                x = tf.concat([x_d, x_c], axis=3)
            else:
                x = tf.concat([x_d, x_c], axis=3)

            # 2 2 1024
            Cn = self.convblock('block1', x, 1024, is_train)
            # 1024
            Cn = tf.reduce_mean(Cn, axis=[1, 2])
            # 1024
            Cn = self.block('block2', Cn, 1024, is_train)
            # 64
            Cn = self.block('block3', Cn, self._latent_size, is_train)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            return Cn

    def block(self, name, x, chout, is_train, is_last=False):
        with tf.variable_scope(name, reuse=self._reuse):
            Cn = ops.linear(x, chout, 'linear')
            if not is_last:
                Cn = ops.norms(Cn, is_train, 'norm', self._norm)
                Cn = ops.acts(Cn, self._activation)
            return Cn

    def convblock(self, name, x, chout, is_train, is_last=False, s=2):
        with tf.variable_scope(name, reuse=self._reuse):
            Cn = ops.conv2d(x, chout, 4, 4, s, s, xavier=True)
            if not is_last:
                Cn = ops.norms(Cn, is_train, 'norm', self._norm)
                Cn = ops.acts(Cn, self._activation)
            return Cn
