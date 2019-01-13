from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.misc import imread, imresize
import numpy as np

import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import session
from tqdm import tqdm
import h5py
from glob import glob


datasets = ['ae_photos', 'apple2orange', 'summer2winter_yosemite', 'horse2zebra',
            'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo',
            'maps', 'cityscapes', 'facades', 'iphone2dslr_flower']


def read_image(path):
    image = imread(path)
    if len(image.shape) != 3 or image.shape[2] != 3:
        print('Wrong image {} with shape {}'.format(path, image.shape))
        return None

    # range of pixel values = [-1.0, 1.0]
    image = image.astype(np.float32) / 255.0
    image = image * 2.0 - 1.0
    return image


def read_images(base_dir):
    ret = []
    for dir_name in ['trainA', 'trainB', 'testA', 'testB']:
        data_dir = os.path.join(base_dir, dir_name)
        paths = glob(os.path.join(data_dir, '*.jpg'))
        print('# images in {}: {}'.format(data_dir, len(paths)))

        images = []
        for path in tqdm(paths):
            image = read_image(path)
            if image is not None:
                images.append(image)
        ret.append((dir_name, images))
    return ret

def store_h5py(base_dir, dir_name, images, image_size):
    print(image_size)
    f = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'w')
    for i in range(len(images)):
        grp = f.create_group(str(i))
        if images[i].shape[0] != image_size:
            image = imresize(images[i], (image_size, image_size, 3))
            # range of pixel values = [-1.0, 1.0]
            image = image.astype(np.float32) / 255.0
            image = image * 2.0 - 1.0
            grp['image'] = image
        else:
            grp['image'] = images[i]
    f.close()


def convert_h5py(task_name):
    print('Generating h5py file')
    base_dir = os.path.join('datasets', task_name)
    data = read_images(base_dir)
    for dir_name, images in data:
        if images[0].shape[0] == 256:
            store_h5py(base_dir, dir_name, images, 256)
        store_h5py(base_dir, dir_name, images, 256)


def read_h5py(task_name, image_size):
    base_dir = 'datasets/' + task_name
    paths = glob(os.path.join(base_dir, '*_{}.hy'.format(image_size)))
    if len(paths) != 4:
        convert_h5py(task_name)
    ret = []
    for dir_name in ['trainA', 'trainB', 'testA', 'testB']:
        try:
            dataset = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'r')
        except:
            raise IOError('Dataset is not available. Please try it again')

        images = []
        for id in dataset:
            images.append(dataset[id]['image'].value.astype(np.float32))
        ret.append(images)
    return ret


def download_dataset(task_name):
    print('Download data %s' % task_name)
    cmd = './download_cyclegan_dataset.sh ' +  task_name
    os.system(cmd)


def get_data(task_name, image_size):
    assert task_name in datasets, 'Dataset {}_{} is not available'.format(
        task_name, image_size)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    base_dir = os.path.join('datasets', task_name)
    print('Check data %s' % base_dir)
    if not os.path.exists(base_dir):
        print('Dataset not found. Start downloading...')
        download_dataset(task_name)
        convert_h5py(task_name)

    print('Load data %s' % task_name)
    train_A, train_B, test_A, test_B = \
        read_h5py(task_name, image_size)
    return train_A, train_B, test_A, test_B


def my_imread(path, is_grayscale=False):
    if is_grayscale:
        return imread(path, flatten=True).astype(np.float)
    else:
        return imread(path).astype(np.float32)


def get_image(image_path, image_size, crop_size=136, crop=True, is_grayscale=False):
    img = my_imread(image_path, is_grayscale)
    if crop:
        img_crop = img[(218-crop_size)//2:(218-crop_size)//2+crop_size, (178-crop_size)//2:(178-crop_size)//2+crop_size, :]
    else:
        img_crop = img
    img = imresize(img_crop, (image_size, image_size, 3))
    img = (img/255.0)*2.0 - 1.0
    return img


# from https://github.com/LynnHo/AttGAN-Tensorflow
def batch_dataset(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset


def disk_image_batch_dataset(img_paths, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def parse_func(path, *label):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3)
        return (img,) + label

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat)

    return dataset


class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

        self._is_eager = tf.executing_eagerly()
        self._eager_iterator = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        if self._is_eager:
            self._eager_iterator = tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        if self._is_eager:
            self._eager_iterator = tfe.Iterator(dataset)
        else:
            self._iterator = dataset.make_initializable_iterator(shared_name='CelebA')
            self.img_iter_init = self._iterator.make_initializer(self._dataset)  # add this..works well
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:
                self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op


class Celeba(Dataset):

    att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    def __init__(self, data_dir, atts, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, sess=None, part='train', crop=True, gpu="0"):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        super(Celeba, self).__init__()

        list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
        if crop:
            img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
            img_dir_png = os.path.join(data_dir, 'img_align_celeba_png')
        else:
            img_dir_jpg = os.path.join(data_dir, 'img_celeba_256_center')
            img_dir_png = os.path.join(data_dir, 'img_celeba_256_center')

        names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
        if os.path.exists(img_dir_png):
            img_paths = [os.path.join(img_dir_png, name.replace('jpg', 'png')) for name in names]
        elif os.path.exists(img_dir_jpg):
            img_paths = [os.path.join(img_dir_jpg, name) for name in names]

        att_id = [Celeba.att_dict[att] + 1 for att in atts]
        labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int64)

        if img_resize == 64:
            # crop as how VAE/GAN do
            offset_h = 41
            offset_w = 21
            img_size = 136
        else:
            offset_h = 41
            offset_w = 21
            img_size = 136

        def _map_func(img, label):
            if crop:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BILINEAR)
            img = tf.add(tf.divide(img, 127.5), -1.0)
            label = (label + 1) // 2
            return img, label

        if part == 'test':
            drop_remainder = False
            shuffle = False
            repeat = 1
            img_paths = img_paths[:10000]
            labels = labels[:10000]
        elif part == 'val':
            img_paths = img_paths[10000:22637]
            labels = labels[10000:22637]
        else:
            img_paths = img_paths[22637:]
            labels = labels[22637:]

        dataset = disk_image_batch_dataset(img_paths=img_paths,
                                           labels=labels,
                                           batch_size=batch_size,
                                           prefetch_batch=prefetch_batch,
                                           drop_remainder=drop_remainder,
                                           map_func=_map_func,
                                           num_threads=num_threads,
                                           shuffle=shuffle,
                                           buffer_size=buffer_size,
                                           repeat=repeat)
        self._bulid(dataset, sess)

        self._img_num = len(img_paths)

    def __len__(self):
        return self._img_num

    @staticmethod
    def check_attribute_conflict(att_batch, att_name, att_names):
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        att_id = att_names.index(att_name)

        for att in att_batch:
            if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] == 1:
                _set(att, 0, 'Bangs')
            elif att_name == 'Bangs' and att[att_id] == 1:
                _set(att, 0, 'Bald')
                _set(att, 0, 'Receding_Hairline')
            elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] == 1:
                for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
            elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] == 1:
                for n in ['Straight_Hair', 'Wavy_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
            elif att_name in ['Mustache', 'No_Beard'] and att[att_id] == 1:
                for n in ['Mustache', 'No_Beard']:
                    if n != att_name:
                        _set(att, 0, n)

        return att_batch


if __name__ == '__main__':
    atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    data = Celeba('./data', atts, 128, 32, part='val')
    batch = data.get_next()
    print(len(data))
    print(batch[1][1], batch[1].dtype)
    print(batch[0].min(), batch[1].max(), batch[0].dtype)
