import argparse
import sys
import signal
import os
from datetime import datetime

import tensorflow as tf
from model import EditableGAN
from utils import logger, makedirs


# parsing cmd arguments
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-t', '--train', action='store_false',
                    help="Training mode")
parser.add_argument('--log_step', default=50, type=int,
                    help="Tensorboard log frequency")
parser.add_argument('--batch_size', default=64, type=int,
                    help="Batch size")
parser.add_argument('--image_size', default=128, type=int,
                    help="Image size")
parser.add_argument('--load_model', default='',
                    help='Model path to load (e.g., train_2017-07-07_01-23-45)')
parser.add_argument('--model_name', default='',
                    help='Set the name of directory to save logs')
parser.add_argument('--epoch', default=60, type=int,
                    help='#epochs to run')
parser.add_argument('--num_labels', default=10, type=int,
                    help="The number of categories")
parser.add_argument('--latent_size', default=64, type=int,
                    help="The number of categories")
parser.add_argument('--disc_iter', default=1, type=int,
                    help="The number of categories")
parser.add_argument('--gpu', type=str, default="0",
                    help='Specify gpu number')
parser.add_argument('--description', default='None', type=str,
                    help='Description for the model')

class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args):
    logger.info('Read data:')

    logger.info('Build graph:')
    model = EditableGAN(args)

    print('######################## GPU ALLOCATION ########################')
    print(args.gpu)
    print('######################## GPU ALLOCATION ########################')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    variables_to_save = tf.global_variables()
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(var_list=variables_to_save, max_to_keep=5)

    logger.info('GLOBAL vars:')
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.get_variable_scope().name)
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    if args.load_model != '':
        model_name = args.load_model
    else:
        model_name = '{}_{}'.format("GAN", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.model_name)

    logdir = './logs'
    makedirs(logdir)
    logdir = os.path.join(logdir, model_name)
    logger.info('Events directory: %s', logdir)
    summary_writer = tf.summary.FileWriter(logdir)

    def init_fn(sess):
        logger.info('Initializing all parameters.')
        sess.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=model.global_step,
                             save_model_secs=1200,
                             save_summaries_secs=30)

    f = open(os.path.join(logdir, 'description.txt'), 'w')
    f.write('Description : \n' + args.description)
    f.close()

    if args.train:
        logger.info("Starting training session.")
        with sv.managed_session() as sess:
            base_dir = os.path.join('results', model_name)
            makedirs(base_dir)
            model.train(sess, summary_writer, base_dir)

    logger.info("Starting testing session.")
    with sv.managed_session() as sess:
        base_dir = os.path.join('results', model_name)
        makedirs(base_dir)
        model.test(sess, base_dir)

def main():
    args, unparsed = parser.parse_known_args()

    def shutdown(signal, frame):
        tf.logging.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    run(args)

if __name__ == "__main__":
    main()
