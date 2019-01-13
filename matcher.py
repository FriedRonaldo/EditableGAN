import tensorflow as tf
from utils import logger
import compare_ops as ops


class Matcher(object):
    def __init__(self, name, norm='batch', activation='tanh', batch_size=512, latent_size=64):
        logger.info('Init Matcher %s', name)
        self.name = name
        self._batch_size = batch_size
        self._latent_size = latent_size
        self._norm = norm
        self._activation = activation
        self._reuse = False

    def __call__(self, z_in, is_train=True):
        with tf.variable_scope(self.name, reuse=self._reuse):
            M = self.block('block1', z_in, 1024, is_train, False, False)
            M = self.block('block2', M, 2048, is_train)
            M = self.block('block3', M, 1024, is_train)
            M = self.block('block4', M, self._latent_size, is_train, False, False)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            return M

    def block(self, name, x, chout, is_train, is_norm=True, is_act=True):
        with tf.variable_scope(name, reuse=self._reuse):
            M = ops.linear(x, chout, 'linear')
            if is_norm:
                M = ops.norms(M, is_train, 'norm', self._norm)
            if is_act:
                M = ops.acts(M, self._activation)

            return M
