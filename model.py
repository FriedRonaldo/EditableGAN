from tqdm import trange
import tensorflow as tf
import numpy as np
import os
import data_loader
import data_saver
import compare_ops as ops

from generator import SNGenerator
from discriminator import SNDiscriminator
from connector import V3CondConnector
from classifier import V3Classifier

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


class EditableGAN(object):
    def __init__(self, args):
        self._log_step = args.log_step
        self._batch_size = args.batch_size
        self._image_size = args.image_size
        self._latent_size = args.latent_size
        self._num_labels = args.num_labels
        self._epochs = args.epoch
        self._d_iter = args.disc_iter

        self._image_shape = [self._image_size, self._image_size, 3]

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.y_b = tf.placeholder(tf.float32, [self._batch_size] + [self._num_labels], 'y_b')
        self.z_input = tf.placeholder(tf.float32, [self._batch_size, self._latent_size], 'z_input')

        self.atts = ['Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
                                                     'Male', 'Mustache', 'Smiling', 'Eyeglasses', 'Young']

        # bang  black   blond   brown   gray    male    mustache     smile   glasses young
        self.tr_data = data_loader.Celeba('../data', self.atts, self._image_size, self._batch_size, num_threads=16,
                                          prefetch_batch=3, gpu=args.gpu, crop=False)
        self.val_data = data_loader.Celeba('../data', self.atts, self._image_size, self._batch_size, num_threads=1,
                                           shuffle=False, part='val', gpu=args.gpu, crop=False)
        self.test_data = data_loader.Celeba('../data', self.atts, self._image_size, self._batch_size, num_threads=1,
                                            shuffle=False, part='test', gpu=args.gpu, crop=False)

        batch = self.tr_data.batch_op

        self.x_a = batch[0]
        self.y_a = batch[1]
        self.y_a = tf.cast(self.y_a, tf.float32)

        G = SNGenerator('generator', image_shape=self._image_shape, batch_size=self._batch_size, norm='batch', activation='lrelu')
        D = SNDiscriminator('discriminator', batch_size=self._batch_size, sn=False, norm='group', activation='lrelu')
        Cn = V3CondConnector('connector', batch_size=self._batch_size, latent_size=self._latent_size, norm='group', activation='lrelu')
        C = V3Classifier('classifier', batch_size=self._batch_size, num_labels=self._num_labels, norm='batch', activation='lrelu')

        self.x_hat_a = G(self.z_input, self.y_a, is_train=self.is_train)

        d_real, cn_dreal = D(self.x_a, self.is_train)
        d_fake, cn_dfake = D(self.x_hat_a, self.is_train)
        c_real, cn_creal, c_real2 = C(self.x_a, self.is_train)
        c_fake, cn_cfake, c_fake2 = C(self.x_hat_a, self.is_train)

        # Vanilla GAN loss
        self.d_adv_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        self.d_adv_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

        self.g_adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

        # Latent vector of x_a
        self.z_x_a, self.cn_y_a = Cn(cn_dreal, cn_creal, self.y_a, self.is_train)
        # Latent vector of x_hat_a = z_input
        self.z_recon, self.cn_y_a_fake = Cn(cn_dfake, cn_cfake, self.y_a, self.is_train)
        # Latent reconstruction loss
        self.cn_recon = tf.reduce_mean(tf.abs(self.z_recon - self.z_input))
        # Condition reconstruction loss
        self.cn_cond = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_a, logits=self.cn_y_a_fake))

        # Reconstructed image
        self.x_a_recon = G(self.z_x_a, self.y_a, self.is_train)
        self.x_hat_recon = G(self.z_recon, self.y_a, self.is_train)

        # Modified image
        self.x_b = G(self.z_x_a, self.y_b, self.is_train)

        offset = self._batch_size // 20.0

        # Classification Loss for y_a
        y_a_sum = tf.reduce_sum(self.y_a, 0)
        pos_w_a = self._batch_size / (2.0 * y_a_sum + offset)
        neg_w_a = self._batch_size / (2.0 * (self._batch_size - y_a_sum) + offset)

        ce_x_hat_a = tf.reduce_sum(ops.weighted_sigmoid_ce_with_logits(self.y_a, c_fake, pos_w_a, neg_w_a), axis=1)
        self.g_ce_a = tf.reduce_mean(ce_x_hat_a)

        ce_x_a = tf.reduce_sum(ops.weighted_sigmoid_ce_with_logits(self.y_a, c_real, pos_w_a, neg_w_a), axis=1)
        self.c_ce = tf.reduce_mean(ce_x_a)

        # Classification Loss for y_b
        y_b_sum = tf.reduce_sum(self.y_b, 0)
        pos_w_b = self._batch_size / (2.0 * y_b_sum + offset)
        neg_w_b = self._batch_size / (2.0 * (self._batch_size - y_a_sum) + offset)

        c_b, cn_cb, c_b2 = C(self.x_b, self.is_train)
        ce_x_b = tf.reduce_sum(ops.weighted_sigmoid_ce_with_logits(self.y_b, c_b, pos_w_b, neg_w_b), axis=1)
        self.g_ce_b = tf.reduce_mean(ce_x_b)

        # Zero-centered GP on real
        self.gradient = tf.gradients(tf.reduce_sum(d_real), [self.x_a])[0]
        self.slope = tf.sqrt(tf.reduce_sum(tf.square(self.gradient), reduction_indices=[1, 2, 3]))
        self.penalty = 10.0 * tf.reduce_mean(tf.square(self.slope))

        # Adv. loss for x_b
        d_b, cn_db = D(self.x_b, self.is_train)
        self.g_adv_b = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_b, labels=tf.ones_like(d_b)))
        self.d_adv_fake_b = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_b, labels=tf.zeros_like(d_b)))

        self.d_adv = self.d_adv_real + (3.0 * self.d_adv_fake + self.d_adv_fake_b) / 4.0
        self.g_adv = (3.0 * self.g_adv + self.g_adv_b) / 4.0

        self.c_adv_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=c_real2, labels=tf.ones_like(c_real2)))
        self.c_adv_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=c_fake2, labels=tf.zeros_like(c_b2)))
        self.c_adv = self.c_adv_real + self.c_adv_fake

        # G
        # adv loss : x_hat_a
        # ce loss : x_hat_a
        # ce loss : x_b
        self.g_loss = self.g_adv + (self.g_ce_a + self.g_ce_b) / 3.0
        # adv loss : x_real / x_fake
        # GP : zero-centered GP
        self.d_loss = self.d_adv + self.penalty
        # Cn
        # reconstruction loss : z_input / z_recon
        # self.cn_loss = self.cn_recon
        self.cn_loss = 2.0 * self.cn_recon + self.cn_cond
        # C
        # ce loss : x_a
        self.c_loss = self.c_ce + self.c_adv

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99).minimize(self.d_loss, var_list=D.var_list)
            self.cn_opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99).minimize(self.cn_loss, var_list=Cn.var_list)
            self.g_opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99).minimize(self.g_loss, var_list=G.var_list, global_step=self.global_step)
            self.c_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.c_loss, var_list=C.var_list)

        # Accuracy
        cp = tf.equal(tf.round(tf.nn.sigmoid(c_real)), self.y_a)
        self.cxi = tf.round(tf.nn.sigmoid(c_real))
        cast_cp = tf.cast(cp, tf.float32)
        acc = tf.reduce_mean(cast_cp)
        self._acc = acc
        each_acc = tf.reduce_mean(cast_cp, 0)
        self.each_acc = each_acc

        tf.summary.scalar('C/loss', self.c_loss)
        tf.summary.scalar('C/acc', acc)
        tf.summary.scalar('C/bang', each_acc[0])
        tf.summary.scalar('C/black', each_acc[1])
        tf.summary.scalar('C/blond', each_acc[2])
        tf.summary.scalar('C/brown', each_acc[3])
        tf.summary.scalar('C/gray', each_acc[4])
        tf.summary.scalar('C/male', each_acc[5])
        tf.summary.scalar('C/mustache', each_acc[6])
        tf.summary.scalar('C/smile', each_acc[7])
        tf.summary.scalar('C/glasses', each_acc[8])
        tf.summary.scalar('C/young', each_acc[9])
        # Accuracy

        tf.summary.scalar('loss/D', self.d_loss)
        tf.summary.scalar('loss/G', self.g_loss)
        tf.summary.scalar('loss/Cn', self.cn_loss)
        tf.summary.scalar('loss/C', self.c_loss)

        tf.summary.scalar('D/adv', self.d_adv)
        tf.summary.scalar('D/penalty', self.penalty)

        tf.summary.scalar('G/adv', self.g_adv)
        tf.summary.scalar('G/ce_a', self.g_ce_a)
        tf.summary.scalar('G/ce_b', self.g_ce_b)

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

        data_size = len(self.tr_data)

        logger.info('   {} images from data'.format(data_size))

        num_batch = data_size // self._batch_size

        initial_step = sess.run(self.global_step)
        num_global_step = self._epochs * num_batch

        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        self.tr_data.img_iter_init.run(session=sess)
        self.val_data.img_iter_init.run(session=sess)

        y_b = ops.generate_y_rand(data_size)

        z_fix = sample_z([self._batch_size, self._latent_size])
        x_fix, y_fix = self.val_data.get_next()
        y_b_fix = ops.generate_y_rand(self._batch_size)

        # Woman & Mustache
        y_b_fix[:self._batch_size//4, 5] = 0
        y_b_fix[:self._batch_size//4, 6] = 1
        # Young & Gray Hair
        y_b_fix[self._batch_size//4:self._batch_size//2, -1] = 1
        y_b_fix[self._batch_size//4:self._batch_size//2, 4] = 1
        y_b_fix[self._batch_size//4:self._batch_size//2, 1] = 0
        y_b_fix[self._batch_size//4:self._batch_size//2, 2] = 0
        y_b_fix[self._batch_size//4:self._batch_size//2, 3] = 0

        # stddev anneal
        stddev_init = 2.0
        stddev = stddev_init

        x_fix_255 = (x_fix + 1) * 127.5
        ssim = [0]
        for step in t:
            epoch = step // num_batch
            iteration = step % num_batch
            if iteration == 0:
                gen_sample, rec_sample, mod_sample = sess.run([self.x_hat_a, self.x_a_recon, self.x_b],
                                                              feed_dict={self.x_a: x_fix,
                                                                         self.y_a: y_fix,
                                                                         self.y_b: y_b_fix,
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
                if epoch > 10:
                    stddev = max(stddev_init - stddev_init * (epoch - 10)/20, 1.0)

            z_batch = sample_z([self._batch_size, self._latent_size], stddev=stddev)
            y_b_batch = y_b[iteration * self._batch_size:(iteration + 1) * self._batch_size]
            sess.run([self.d_opt, self.g_opt, self.cn_opt, self.c_opt], feed_dict={self.z_input: z_batch,
                                                                                   self.y_b: y_b_batch,
                                                                                   self.is_train: True})

            fetches = [self.d_loss, self.g_loss, self.cn_loss, self.c_loss]

            if step % self._log_step == 0:
                fetches += [self.summary_op]
                t.set_description('\n')
                x_val, y_val = self.val_data.get_next()
                fetched = sess.run(fetches, feed_dict={self.x_a: x_val,
                                                       self.y_a: y_val,
                                                       self.y_b: y_b_batch,
                                                       self.z_input: z_batch,
                                                       self.is_train: False
                                                       })
                summary_writer.add_summary(fetched[-1], step)
                summary_writer.flush()
                t.set_description(
                    'Epoch ({}) iter({}) loss: D({:.3f}) G({:.3f}) Cn({:.3f}) C({:.3f}) ssim({:.3f})'.format(
                        (epoch + 1), iteration, fetched[0], fetched[1], fetched[2], fetched[3], ssim[0]))

    def test(self, sess, base_dir):
        self.val_data.img_iter_init.run(session=sess)
        self.test_data.img_iter_init.run(session=sess)

        num_val = len(self.val_data)
        num_test = len(self.test_data)

        print('{} data from validation\n{} data from test'.format(num_val, num_test))

        # Select one ( not tested for multiple tasks ... self.val_data and self.test_data may be re-initialized)

        ssim = False

        modification = False

        latent_partial = False

        generation = False

        latent_flip = False

        latent_flip_to_np = False

        ############################
        #           SSIM           #
        ############################
        if ssim:
            print('Start Quantitative Evaluation')

            t_val = trange(0, num_val, total=num_val, initial=0)
            t_test = trange(0, num_test, total=num_test, initial=0)

            val_ssim = []

            t_val.set_description('VALIDATION')

            for i in t_val:
                x_val, y_val = self.val_data.get_next()
                val_rec = sess.run([self.x_a_recon], feed_dict={self.x_a: x_val,
                                                                self.y_a: y_val,
                                                                self.is_train: False})
                val_rec = val_rec[0]
                x_val_255 = (x_val + 1.0) * 127.5
                rec_255 = (val_rec + 1.0) * 127.5
                tmp = ops.eval_ssim(x_val_255, rec_255)
                val_ssim.append((i, tmp[0]))

            test_ssim = []

            t_test.set_description('TEST')

            for i in t_test:
                x_test, y_test = self.test_data.get_next()
                test_rec = sess.run([self.x_a_recon], feed_dict={self.x_a: x_test,
                                                                 self.y_a: y_test,
                                                                 self.is_train: False})
                test_rec = test_rec[0]
                x_test_255 = (x_test + 1.0) * 127.5
                rec_255 = (test_rec + 1.0) * 127.5
                tmp = ops.eval_ssim(x_test_255, rec_255)
                test_ssim.append((i, tmp[0]))

            val_sorted = sorted(val_ssim, key=lambda x: x[1], reverse=True)
            test_sorted = sorted(test_ssim, key=lambda x: x[1], reverse=True)

            file_val = open(os.path.join(base_dir, 'validation_ssim_sorted.txt'), 'w')
            file_val.write('INDEX\tSSIM\n')
            file_test = open(os.path.join(base_dir, 'test_ssim_sorted.txt'), 'w')
            file_test.write('INDEX\tSSIM\n')

            val_mean = tuple(map(np.mean, zip(*val_sorted)))
            test_mean = tuple(map(np.mean, zip(*test_sorted)))

            print('VALIDATION MEAN SSIM', val_mean)
            print('TEST MEAN SSIM', test_mean)

            for v in val_sorted:
                file_val.write('{:05d}\t{:.4f}\n'.format(v[0], v[1]))

            for t in test_sorted:
                file_test.write('{:05d}\t{:.4f}\n'.format(t[0], t[1]))

        ############################
        #       Modification       #
        ############################
        if modification:
            print('Start Qualitative Evaluation')

            val_dir = os.path.join(base_dir, 'qualitative', 'validation')
            test_dir = os.path.join(base_dir, 'qualitative', 'test')

            if not os.path.exists(val_dir):
                os.makedirs(val_dir)

            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            t_val = trange(0, num_val, total=num_val, initial=0)
            t_test = trange(0, num_test, total=num_test, initial=0)

            # original / reconstructed / modified //
            # invert the attribute , ex) mustache -> non-mustache // For hair color : not invert but change to $color
            # bang  black   blond   brown   gray    male    mustache     smile   glasses young
            t_val.set_description('VALIDATION')
            for i in t_val:
                x_val, y_val = self.val_data.get_next()
                returned = np.copy(x_val)
                # modify one by one
                for k in range(self._num_labels):
                    y_mod = np.copy(y_val)
                    # 1~4 : hair color, exclusively applied .. can be applied simultaneously - New color
                    if k == 1:
                        y_mod[:, 1] = 1
                        y_mod[:, 2] = 0
                        y_mod[:, 3] = 0
                        y_mod[:, 4] = 0
                    elif k==2:
                        y_mod[:, 1] = 0
                        y_mod[:, 2] = 1
                        y_mod[:, 3] = 0
                        y_mod[:, 4] = 0
                    elif k==3:
                        y_mod[:, 1] = 0
                        y_mod[:, 2] = 0
                        y_mod[:, 3] = 1
                        y_mod[:, 4] = 0
                    elif k==4:
                        y_mod[:, 1] = 0
                        y_mod[:, 2] = 0
                        y_mod[:, 3] = 0
                        y_mod[:, 4] = 1
                    else:
                        y_mod[:, k] = 1 - y_val[:, k]

                    if k == 0:
                        rec, mod = sess.run([self.x_a_recon, self.x_b], feed_dict={self.x_a: x_val,
                                                                                   self.y_a: y_val,
                                                                                   self.y_b: y_mod,
                                                                                   self.is_train: False})
                        rec = np.reshape(rec, [self._batch_size] + self._image_shape)
                        mod = np.reshape(mod, [self._batch_size] + self._image_shape)
                        returned = np.concatenate([returned, rec, mod], axis=2)
                    else:
                        mod = sess.run([self.x_b], feed_dict={self.x_a: x_val,
                                                              self.y_a: y_val,
                                                              self.y_b: y_mod,
                                                              self.is_train: False})
                        mod = np.reshape(mod, [self._batch_size] + self._image_shape)
                        returned = np.concatenate([returned, mod], axis=2)
                data_saver.save_images(returned, [1, 1], os.path.join(val_dir, '{:05d}.png'.format(i)))

            t_test.set_description('TEST')
            for i in t_test:
                x_test, y_test = self.test_data.get_next()
                returned = np.copy(x_test)
                # modify one by one
                for k in range(self._num_labels):
                    y_mod = np.copy(y_test)
                    # 1~4 : hair color, exclusively applied .. can be applied simultaneously - New color
                    if k == 1:
                        y_mod[:, 1] = 1
                        y_mod[:, 2] = 0
                        y_mod[:, 3] = 0
                        y_mod[:, 4] = 0
                    elif k==2:
                        y_mod[:, 1] = 0
                        y_mod[:, 2] = 1
                        y_mod[:, 3] = 0
                        y_mod[:, 4] = 0
                    elif k==3:
                        y_mod[:, 1] = 0
                        y_mod[:, 2] = 0
                        y_mod[:, 3] = 1
                        y_mod[:, 4] = 0
                    elif k==4:
                        y_mod[:, 1] = 0
                        y_mod[:, 2] = 0
                        y_mod[:, 3] = 0
                        y_mod[:, 4] = 1
                    else:
                        y_mod[:, k] = 1 - y_test[:, k]
                    if k == 0:
                        rec, mod = sess.run([self.x_a_recon, self.x_b], feed_dict={self.x_a: x_test,
                                                                                   self.y_a: y_test,
                                                                                   self.y_b: y_mod,
                                                                                   self.is_train: False})
                        rec = np.reshape(rec, [self._batch_size] + self._image_shape)
                        mod = np.reshape(mod, [self._batch_size] + self._image_shape)
                        returned = np.concatenate([returned, rec, mod], axis=2)
                    else:
                        mod = sess.run([self.x_b], feed_dict={self.x_a: x_test,
                                                              self.y_a: y_test,
                                                              self.y_b: y_mod,
                                                              self.is_train: False})
                        mod = np.reshape(mod, [self._batch_size] + self._image_shape)
                        returned = np.concatenate([returned, mod], axis=2)
                data_saver.save_images(returned, [1, 1], os.path.join(test_dir, '{:05d}.png'.format(i)))

        ##################################
        # Latent space walking partially #
        ##################################
        if latent_partial:
            # Partially walking along the latent space
            unit = 8
            latent_dir = os.path.join(base_dir, 'experiment', 'latent', 'partial{}'.format(unit))

            if not os.path.exists(latent_dir):
                os.makedirs(latent_dir)

            part = self._latent_size // unit

            t_latent = trange(0, 10000//part, initial=0, total=10000//part)

            # For each sample
            for i in t_latent:
                z_a = sample_z([self._batch_size, self._latent_size], stddev=0.8)
                z_b = sample_z([self._batch_size, self._latent_size], stddev=0.8)
                y_a = ops.generate_y_rand(self._batch_size)
                y_b = ops.generate_y_rand(self._batch_size)

                # For each part
                x_one_z = None
                for j in range(part):
                    x_part = None
                    z_inter = np.copy(z_a)
                    for k in range(12):
                        z_inter[:, j*unit:(j+1)*unit] = np.copy((1.0 - 0.1 * k) * z_a[:, j * unit:(j+1) * unit] \
                                                        + 0.1 * k * z_b[:, j * unit:(j+1) * unit])
                        y = np.copy(y_a)
                        if k == 11:
                            z_inter = np.copy(z_b)
                            y = np.copy(y_b)
                        x_ = sess.run([self.x_hat_a], feed_dict={self.z_input: z_inter,
                                                                 self.y_a: y,
                                                                 self.is_train: False})
                        x_ = np.reshape(x_, [self._batch_size] + self._image_shape)
                        if k == 0:
                            x_part = np.copy(x_)
                        else:
                            x_part = np.concatenate([x_part, x_], axis=2)
                    if j == 0:
                        x_one_z = np.copy(x_part)
                    else:
                        x_one_z = np.concatenate([x_one_z, x_part], axis=1)
                data_saver.save_images(x_one_z, [1, 1], os.path.join(latent_dir, '{:05d}.png'.format(i)))

        #############################
        # Latent space walking flip #
        #############################
        if latent_flip:
            latent_dir_val = os.path.join(base_dir, 'experiment', 'latent', 'flip', 'validation')
            latent_dir_test = os.path.join(base_dir, 'experiment', 'latent', 'flip', 'test')

            if not os.path.exists(latent_dir_val):
                os.makedirs(latent_dir_val)

            if not os.path.exists(latent_dir_test):
                os.makedirs(latent_dir_test)

            t_val = trange(0, num_val, total=num_val, initial=0)
            t_test = trange(0, num_test, total=num_test, initial=0)

            t_val.set_description('VALIDATION')

            for i in t_val:
                x_left, y_ = self.val_data.get_next()
                x_right = np.flip(x_left, axis=2)
                z_left = sess.run([self.z_x_a], feed_dict={self.x_a: x_left,
                                                           self.y_a: y_,
                                                           self.is_train: False})
                z_right = sess.run([self.z_x_a], feed_dict={self.x_a: x_right,
                                                           self.y_a: y_,
                                                           self.is_train: False})
                returned = np.copy(x_left)
                for k in range(11):
                    z_inter = np.copy((1.0 - 0.1 * k) * z_left[0] + 0.1 * k * z_right[0])
                    y_inter = np.copy(y_)
                    x_ = sess.run([self.x_hat_a], feed_dict={self.z_input: z_inter,
                                                             self.y_a: y_inter,
                                                             self.is_train: False})
                    x_ = np.reshape(x_, [self._batch_size] + self._image_shape)
                    returned = np.concatenate([returned, x_], axis=2)
                returned = np.concatenate([returned, x_right], axis=2)
                data_saver.save_images(returned, [1, 1], os.path.join(latent_dir_val, '{:05d}.png'.format(i)))

            t_test.set_description('TEST')

            for i in t_test:
                x_left, y_ = self.test_data.get_next()
                x_right = np.flip(x_left, axis=2)
                z_left = sess.run([self.z_x_a], feed_dict={self.x_a: x_left,
                                                           self.y_a: y_,
                                                           self.is_train: False})
                z_right = sess.run([self.z_x_a], feed_dict={self.x_a: x_right,
                                                            self.y_a: y_,
                                                            self.is_train: False})
                returned = np.copy(x_left)
                for k in range(11):
                    z_inter = np.copy((1.0 - 0.1 * k) * z_left[0] + 0.1 * k * z_right[0])
                    y_inter = np.copy(y_)
                    x_ = sess.run([self.x_hat_a], feed_dict={self.z_input: z_inter,
                                                             self.y_a: y_inter,
                                                             self.is_train: False})
                    x_ = np.reshape(x_, [self._batch_size] + self._image_shape)
                    returned = np.concatenate([returned, x_], axis=2)
                returned = np.concatenate([returned, x_right], axis=2)
                data_saver.save_images(returned, [1, 1], os.path.join(latent_dir_test, '{:05d}.png'.format(i)))

        #############################
        # Latent space walking flip #
        #############################
        if latent_flip_to_np:
            latent_dir_val = os.path.join(base_dir, 'experiment', 'latent', 'numpy', 'validation')
            latent_dir_test = os.path.join(base_dir, 'experiment', 'latent', 'numpy', 'test')

            if not os.path.exists(latent_dir_val):
                os.makedirs(latent_dir_val)

            if not os.path.exists(latent_dir_test):
                os.makedirs(latent_dir_test)

            t_val = trange(0, num_val, total=num_val, initial=0)
            t_test = trange(0, num_test, total=num_test, initial=0)

            t_val.set_description('VALIDATION')
            t_test.set_description('TEST')

            np_left_val = None
            np_right_val = None

            for i in t_val:
                x_left, y_ = self.val_data.get_next()
                x_right = np.flip(x_left, axis=2)
                z_left = sess.run([self.z_x_a], feed_dict={self.x_a: x_left,
                                                           self.y_a: y_,
                                                           self.is_train: False})
                z_right = sess.run([self.z_x_a], feed_dict={self.x_a: x_right,
                                                            self.y_a: y_,
                                                            self.is_train: False})
                if i == 0:
                    np_left_val = np.copy(z_left)
                    np_right_val = np.copy(z_right)

                else:
                    np_left_val = np.concatenate([np_left_val, z_left], axis=0)
                    np_right_val = np.concatenate([np_right_val, z_right], axis=0)

            np_left_test = None
            np_right_test = None

            for i in t_test:
                x_left, y_ = self.test_data.get_next()
                x_right = np.flip(x_left, axis=2)
                z_left = sess.run([self.z_x_a], feed_dict={self.x_a: x_left,
                                                           self.y_a: y_,
                                                           self.is_train: False})
                z_right = sess.run([self.z_x_a], feed_dict={self.x_a: x_right,
                                                            self.y_a: y_,
                                                            self.is_train: False})
                if i == 0:
                    np_left_test = np.copy(z_left)
                    np_right_test = np.copy(z_right)

                else:
                    np_left_test = np.concatenate([np_left_test, z_left], axis=0)
                    np_right_test = np.concatenate([np_right_test, z_right], axis=0)

            np.save(os.path.join(latent_dir_val, 'left_val'), np_left_val)
            np.save(os.path.join(latent_dir_val, 'right_val'), np_right_val)
            np.save(os.path.join(latent_dir_test, 'left_test'), np_left_test)
            np.save(os.path.join(latent_dir_test, 'right_test'), np_right_test)

        ##############
        # Generation #
        ##############
        if generation:
            generation_dir_rand = os.path.join(base_dir, 'experiment', 'generation', 'y_random')
            generation_dir_train = os.path.join(base_dir, 'experiment', 'generation', 'y_train')

            if not os.path.exists(generation_dir_rand):
                os.makedirs(generation_dir_rand)

            if not os.path.exists(generation_dir_train):
                os.makedirs(generation_dir_train)

            t_rand = trange(0, 200, total=200, initial=0)
            t_train = trange(0, 200, total=200, initial=0)

            img_frame_dim = int(np.floor(np.sqrt(self._batch_size)))

            stddev = 1.0

            for i in t_rand:
                z_batch = sample_z([self._batch_size, self._latent_size], stddev=stddev)
                y_batch = ops.generate_y_rand(self._batch_size)
                x_hat = sess.run(self.x_hat_a, feed_dict={self.z_input: z_batch,
                                                          self.y_a: y_batch,
                                                          self.is_train: False})

                data_saver.save_images(x_hat, [img_frame_dim, img_frame_dim], os.path.join(generation_dir_rand, '{:05d}.png'.format(i)))

            for i in t_train:
                z_batch = sample_z([self._batch_size, self._latent_size], stddev=stddev)
                x_hat = sess.run(self.x_hat_a, feed_dict={self.z_input: z_batch,
                                                          self.is_train: False})

                data_saver.save_images(x_hat, [img_frame_dim, img_frame_dim], os.path.join(generation_dir_train, '{:05d}.png'.format(i)))