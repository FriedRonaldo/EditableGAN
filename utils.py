import logging
import os
import tensorflow as tf

# start logging
logging.info("Start GAN")
logger = logging.getLogger('GAN')
logger.setLevel(logging.INFO)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """Return a Session with simple config."""
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)
