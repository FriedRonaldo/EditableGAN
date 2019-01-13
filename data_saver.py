import numpy as np
import scipy.misc
# import itertools
# from matplotlib import pyplot as plt
import glob
from skimage.transform import resize
from skimage import io, img_as_float
import os


# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py
def merge(imgs, size):
    h, w = imgs.shape[1], imgs.shape[2]
    if imgs.shape[3]==1:
        img = np.zeros((h*size[0], w*size[1]))
        for idx, im in enumerate(imgs):
            i = idx%size[1]
            j = idx//size[1]

            img[j*h:j*h+h, i*w:i*w+w] = im[:, :, 0]

    elif imgs.shape[3] in (3, 4):
        c = imgs.shape[3]
        img = np.zeros((h*size[0], w*size[1], c))
        for idx, im in enumerate(imgs):
            i = idx%size[1]
            j = idx//size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = im

    return img


def imsave(imgs, size, path):
    img = (merge(imgs, size))
    return scipy.misc.imsave(path, img)


def inverse_transform(images):
    return (images+1.0)/2.0


def save_images(imgs, size, image_path):
    return imsave(inverse_transform(imgs), size, image_path)


def save_images_wo_inverse(imgs, size, image_path):
    return imsave(imgs, size, image_path)

#
# def show_images(size_figure_grid, samples, img_shape = (28, 28)):
#     fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
#     for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)
#
#     for k in range(size_figure_grid * size_figure_grid):
#         i = k // size_figure_grid
#         j = k % size_figure_grid
#         ax[i, j].cla()
#         if img_shape == (28, 28):
#             ax[i, j].imshow(np.reshape(samples[k], img_shape), cmap='gray')
#         else:
#             ax[i, j].imshow(np.reshape(samples[k], img_shape))#, cmap='gray')
#
#     plt.show()


def resize_images(image_path, size=128):
    """
    :param image_path: query images path
    :param size: size to resize
    Save resized images to ./image_path/resized_image/
    """
    if not os.path.exists(image_path+'/resized_image'):
        os.mkdir(image_path+'/resized_image')
    files = glob.glob(image_path+'/*.jpg')
    i = 0
    for f in files:
        tmp = img_as_float(io.imread(f))
        tmp_res = resize(tmp, (size, size))
        io.imsave(image_path+'/resized_image/'+str(i)+'.jpg', tmp_res)
        i += 1