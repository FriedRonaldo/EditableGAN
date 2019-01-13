from tqdm import trange
import tensorflow as tf
import numpy as np
import os
import math
from scipy.ndimage import rotate

import data_loader
import data_saver
import compare_ops as ops

from tensorpack import *
from tensorpack.dataflow import dataset

from generator import MNISTGenerator
from discriminator import MNISTDiscriminator
from connector import MNISTConnector
from classifier import MNISTClassifier

from utils import logger


def sample_z(size, type='normal', p=0.5, stddev=1.0):
    assert type in ['normal', 'uniform', 'binomial']
    if type == 'normal':
        return np.random.normal(size=size, scale=stddev).astype(np.float32)
    elif type == 'uniform':
        return np.random.uniform(low=-1.0, high=1.0, size=size).astype(np.float32)
    elif type == 'binomial':
        return np.random.binomial(1, p, size=size).astype(np.float32)
    else:
        return np.random.normal(size=size).astype(np.float32)


def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'
    ds = dataset.Mnist(train_or_test, isTrain)
    datasize = len(ds)
    if isTrain:
        augmentors = [
            imgaug.Rotation(45.0)
        ]
    else:
        augmentors = [
            imgaug.Rotation(45.0)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds, datasize


class MNIST(object):
    def __init__(self, args):
        self._log_step = args.log_step
        self._batch_size = args.batch_size
        self._image_size = args.image_size
        self._latent_size = args.latent_size
        self._num_labels = args.num_labels
        self._epochs = args.epoch

        self._image_shape = [self._image_size, self._image_size, 1]

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.x_a = tf.placeholder(tf.float32, shape=[self._batch_size] + self._image_shape, name='x_input')
        self.y_input = tf.placeholder(tf.int32, shape=[self._batch_size], name='y_input')
        self.y_b_in = tf.placeholder(tf.int32, [self._batch_size], 'y_b_in')
        self.z_input = tf.placeholder(tf.float32, [self._batch_size, self._latent_size], 'z_input')

        self.y_a = tf.to_float(tf.reshape(tf.one_hot(self.y_input, 10), [self._batch_size, 10]))
        self.y_b = tf.to_float(tf.reshape(tf.one_hot(self.y_b_in, 10), [self._batch_size, 10]))

        G = MNISTGenerator('generator', image_shape=self._image_shape, batch_size=self._batch_size, norm='batch', activation='relu')
        D = MNISTDiscriminator('discriminator', batch_size=self._batch_size, sn=False, norm='batch', activation='lrelu')
        Cn = MNISTConnector('connector', batch_size=self._batch_size, latent_size=self._latent_size, norm='batch', activation='lrelu')
        C = MNISTClassifier('classifier', batch_size=self._batch_size, num_labels=self._num_labels, norm='batch', activation='lrelu')

        self.x_hat_a = G(self.z_input, self.y_a, is_train=self.is_train)

        d_real, cn_dreal = D(self.x_a, self.is_train)
        d_fake, cn_dfake = D(self.x_hat_a, self.is_train)
        c_real, cn_creal = C(self.x_a, self.is_train)
        c_fake, cn_cfake = C(self.x_hat_a, self.is_train)

        # Vanilla GAN loss
        self.d_adv_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        self.d_adv_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

        self.g_adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

        # Latent vector of x_a
        self.z_x_a = Cn(cn_dreal, cn_creal, self.y_a, self.is_train)
        # Latent vector of x_hat_a = z_input
        self.z_recon = Cn(cn_dfake, cn_cfake, self.y_a, self.is_train)
        # Latent reconstruction loss
        self.cn_recon = tf.reduce_mean(tf.abs(self.z_recon - self.z_input))

        # Reconstructed image
        self.x_a_recon = G(self.z_x_a, self.y_a, self.is_train)
        self.x_hat_recon = G(self.z_recon, self.y_a, self.is_train)

        # Modified image
        self.x_b = G(self.z_x_a, self.y_b, self.is_train)

        # classification loss
        self.g_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_a, logits=c_fake))
        self.c_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_a, logits=c_real))

        self.d_adv = self.d_adv_real + self.d_adv_fake
        self.g_adv = self.g_adv

        # G
        # adv loss : x_hat_a
        # ce loss : x_hat_a
        self.g_loss = self.g_adv + self.g_ce
        # adv loss : x_real / x_fake
        self.d_loss = self.d_adv
        # Cn
        # reconstruction loss : z_input / z_recon
        # self.cn_loss = self.cn_recon
        self.cn_loss = self.cn_recon
        # C
        # ce loss : x_a
        self.c_loss = self.c_ce

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=D.var_list)
            self.cn_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.cn_loss, var_list=Cn.var_list)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=G.var_list, global_step=self.global_step)
            self.c_opt = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(self.c_loss, var_list=C.var_list)

        # Accuracy
        cp = tf.equal(tf.round(tf.nn.softmax(c_real)), self.y_a)
        cast_cp = tf.cast(cp, tf.float32)
        acc = tf.reduce_mean(cast_cp)
        self._acc = acc

        # Accuracy

        tf.summary.scalar('loss/D', self.d_loss)
        tf.summary.scalar('loss/G', self.g_loss)
        tf.summary.scalar('loss/Cn', self.cn_loss)
        tf.summary.scalar('loss/C', self.c_loss)

        tf.summary.scalar('D/adv', self.d_adv)

        tf.summary.scalar('G/adv', self.g_adv)
        tf.summary.scalar('G/ce_a', self.g_ce)

        tf.summary.scalar('C/ce', self.c_ce)

        tf.summary.scalar('Cn/recon_a', self.cn_recon)

        tf.summary.image('image/x_a', self.x_a)
        tf.summary.image('image/x_a_recon', self.x_a_recon)
        tf.summary.image('image/x_hat_a', self.x_hat_a)
        tf.summary.image('image/x_hat_a_recon', self.x_hat_recon)
        tf.summary.image('image/x_b', self.x_b)

        tf.summary.histogram('hist/z_input', self.z_input)
        tf.summary.histogram('hist/z_x_a', self.z_x_a)
        tf.summary.histogram('hist/z_recon', self.z_recon)

        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, base_dir):
        logger.info('Start training.')

        data_train, size_train = get_data('train', self._batch_size)
        data_test, size_test = get_data('test', self._batch_size)

        num_batch = size_train // self._batch_size
        num_batch_val = size_test // self._batch_size

        logger.info('   {} images from data'.format(data_train))

        initial_step = sess.run(self.global_step)
        num_global_step = self._epochs * num_batch

        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        train_flow = data_train.get_data()
        test_flow = data_test.get_data()

        x_fix, y_fix = next(test_flow)

        x_fix = (np.reshape(x_fix, [self._batch_size] + self._image_shape) - 0.5) * 2.0

        y_b_fix = np.copy(y_fix)
        np.random.shuffle(y_b_fix)
        print(y_fix.shape)
        print(y_b_fix.shape)

        z_fix = sample_z([self._batch_size, self._latent_size])

        x_fix_255 = (x_fix + 1) * 127.5
        ssim = [0]
        for step in t:
            epoch = step // num_batch
            iteration = step % num_batch
            if iteration == 0:
                train_flow = data_train.get_data()
                gen_sample, rec_sample, mod_sample = sess.run([self.x_hat_a, self.x_a_recon, self.x_b],
                                                              feed_dict={self.x_a: x_fix,
                                                                         self.y_input: y_fix,
                                                                         self.y_b_in: y_b_fix,
                                                                         self.z_input: z_fix,
                                                                         self.is_train: False})
                img_frame_dim = int(np.floor(np.sqrt(self._batch_size)))
                gen_sample = np.reshape(gen_sample, [-1] + self._image_shape)
                rec_sample = np.reshape(rec_sample, [-1] + self._image_shape)
                mod_sample = np.reshape(mod_sample, [-1] + self._image_shape)
                data_saver.save_images(gen_sample[: img_frame_dim ** 2, :, :, :], [img_frame_dim, img_frame_dim],
                                       os.path.join(base_dir, str(epoch) + '_gen.jpg'))
                data_saver.save_images(rec_sample[: img_frame_dim ** 2, :, :, :], [img_frame_dim, img_frame_dim],
                                       os.path.join(base_dir, str(epoch) + '_rec.jpg'))
                data_saver.save_images(mod_sample[: img_frame_dim ** 2, :, :, :], [img_frame_dim, img_frame_dim],
                                       os.path.join(base_dir, str(epoch) + '_mod.jpg'))
                if epoch == 0:
                    data_saver.save_images(x_fix[: img_frame_dim ** 2, :, :, :], [img_frame_dim, img_frame_dim],
                                           os.path.join(base_dir, str(epoch) + '_orig.jpg'))
                recon_sample = (rec_sample + 1) * 127.5
                ssim = ops.eval_ssim(x_fix_255, recon_sample)
                print(ssim)

            z_batch = sample_z([self._batch_size, self._latent_size])
            x_batch, y_batch = next(train_flow)
            x_batch = (x_batch - 0.5) * 2
            x_batch = np.reshape(x_batch, [self._batch_size] + self._image_shape)
            y_b_batch = np.copy(y_batch)
            np.random.shuffle(y_b_batch)

            sess.run([self.d_opt, self.g_opt, self.cn_opt, self.c_opt], feed_dict={self.z_input: z_batch,
                                                                                   self.x_a: x_batch,
                                                                                   self.y_input: y_batch,
                                                                                   self.y_b_in: y_b_batch,
                                                                                   self.is_train: True})

            fetches = [self.d_loss, self.g_loss, self.cn_loss, self.c_loss]

            if step % self._log_step == 0:
                val_iter = step % num_batch_val
                if val_iter == 0:
                    test_flow = data_test.get_data()
                fetches += [self.summary_op]
                t.set_description('\n')

                x_val, y_val = next(test_flow)
                x_val = np.reshape(x_val, [self._batch_size] + self._image_shape)
                x_val = (x_val - 0.5) * 2
                fetched = sess.run(fetches, feed_dict={self.x_a: x_val,
                                                       self.y_input: y_val,
                                                       self.y_b_in: y_b_batch,
                                                       self.z_input: z_batch,
                                                       self.is_train: False
                                                       })
                summary_writer.add_summary(fetched[-1], step)
                summary_writer.flush()
                t.set_description(
                    'Epoch ({}) iter({}) loss: D({:.3f}) G({:.3f}) Cn({:.3f}) C({:.3f}) ssim({:.3f})'.format(
                        (epoch + 1), iteration, fetched[0], fetched[1], fetched[2], fetched[3], ssim[0]))

    def test(self, sess, base_dir):
        data_test, size_test = get_data('test', self._batch_size)
        num_batch_val = size_test // self._batch_size

        latent_walk = True

        modification = False

        generation = False

        if latent_walk:
            test_flow = data_test.get_data()

            latent_dir = os.path.join(base_dir, 'Interpolation', 'Rotation')

            if not os.path.exists(latent_dir):
                os.makedirs(latent_dir)

            print('Start Latent space walking')

            t_latent = trange(0, num_batch_val, total=num_batch_val, initial=0)

            for i in t_latent:
                x_, y_ = next(test_flow)
                x_l = rotate(np.copy(x_), 45.0, axes=(1, 2), reshape=False, mode='nearest', order=1)
                x_r = rotate(np.copy(x_), -45.0, axes=(1, 2), reshape=False, mode='nearest', order=1)
                x_l = (np.expand_dims(x_l, axis=3) - 0.5) * 2.0
                x_r = (np.expand_dims(x_r, axis=3) - 0.5) * 2.0
                res = np.copy(x_l)
                z_l = sess.run([self.z_x_a], feed_dict={self.x_a: x_l,
                                                        self.y_input: y_,
                                                        self.is_train: False})
                z_r = sess.run([self.z_x_a], feed_dict={self.x_a: x_r,
                                                        self.y_input: y_,
                                                        self.is_train: False})
                for j in range(11):
                    z_inter = (1.0 - 0.1 * j) * z_l[0] + 0.1 * j * z_r[0]
                    x_inter = sess.run([self.x_hat_a], feed_dict={self.z_input: z_inter,
                                                                  self.y_input: y_,
                                                                  self.is_train: False})
                    x_inter = np.reshape(x_inter, [self._batch_size] + self._image_shape)
                    res = np.concatenate([res, x_inter], axis=2)
                res = np.concatenate([res, x_r], axis=2)

                data_saver.save_images(res, [1, 1], os.path.join(latent_dir, '{:05d}.png'.format(i)))

        if modification:
            test_flow = data_test.get_data()

            modification_dir = os.path.join(base_dir, 'Modification')

            if not os.path.exists(modification_dir):
                os.makedirs(modification_dir)

            print('Start Modification')

            t_modification = trange(0, num_batch_val, total=num_batch_val, initial=0)

            for i in t_modification:
                x_, y_ = next(test_flow)
                x_ = (x_ - 0.5) * 2.0
                x_ = np.expand_dims(x_, axis=3)
                y_orig = np.ones(shape=[self._batch_size,], dtype=np.int32)
                res = np.copy(x_)
                z_ = sess.run([self.z_x_a], feed_dict={self.x_a: x_,
                                                       self.y_input: y_,
                                                       self.is_train: False})

                for j in range(10):
                    y_tmp = j * y_orig
                    x_mod = sess.run([self.x_b], feed_dict={self.z_x_a: z_[0],
                                                            self.y_input: y_tmp,
                                                            self.y_b_in: y_tmp,
                                                            self.is_train: False})
                    x_mod = np.reshape(x_mod, [self._batch_size] + self._image_shape)
                    res = np.concatenate([res, x_mod], axis=2)

                data_saver.save_images(res, [1, 1], os.path.join(modification_dir, '{:05d}.png'.format(i)))

        if generation:
            test_flow = data_test.get_data()

            generation_dir = os.path.join(base_dir, 'Generation')

            if not os.path.exists(generation_dir):
                os.makedirs(generation_dir)

            print('Start Generation')

            t_generation = trange(0, num_batch_val, total=num_batch_val, initial=0)

            for i in t_generation:
                pass