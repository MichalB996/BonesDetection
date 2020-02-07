import os
from skimage import transform, io
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

#from load_data import loadData, loadDataOrig
from model.losses import bce_dice_loss, dice_coeff
from keras.optimizers import RMSprop

import numpy as np

from keras.models import model_from_json

from skimage import morphology, color, io, exposure

from skimage import img_as_ubyte

import os.path  # from os.path import exists
import shutil  # shutil.copy2

import cv2

import argparse

data_bow_legs_dir = 'data_bow-legs'
dataset_bow_legs_dir = 'dataset_bow-legs'

import keras.backend as K

import pandas as pd


def binary_crossentropy_custom(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red,
	predicted lung field filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))

    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt

    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute(image, image_path, mask, mask_path, filename_result):
    # Load test data
    im_shape = (512, 256)
    csv_path = r'C:\WD_moje\idx_test.csv'
    df = pd.read_csv(csv_path)

    X, y = [], []

    img = transform.resize(image, im_shape, mode='constant')
    img = np.expand_dims(img, -1)
    mask = transform.resize(mask, im_shape, mode='constant')
    mask = np.expand_dims(mask, -1)
    X.append(img)
    y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_weights = "models/trained_model.hdf5"
    json_file = open('models/model_bk.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(model_weights)

    UNet = loaded_model
    model = loaded_model
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    i = 0
    num_imgs = X.shape[0]
    for ii in range(num_imgs):
        xx_ = X[ii, :, :, :]
        yy_ = y[ii, :, :, :]
        xx = xx_[None, ...]
        yy = yy_[None, ...]
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        mask = yy[..., 0].reshape(inp_shape[:2])

        gt = mask > 0.5
        pr = pred > 0.5

        pr_bin = img_as_ubyte(pr)
        pr_openned = morphology.opening(pr_bin)

        pr = remove_small_regions(pr, 0.005 * np.prod(im_shape))

        sub_dir_file_name = df.iloc[i][0]
        file_name = sub_dir_file_name[9:]
        sub_dir_name = sub_dir_file_name[:8]

        file_name_no_ext = os.path.splitext(file_name)[0]
        file_name_in = mask_path

        im_name_x_ray_original_size = image_path
        im_name_x_ray_original_size_test = image_path

        im_x_ray_original_size = cv2.imread(im_name_x_ray_original_size_test, cv2.IMREAD_GRAYSCALE)
        if im_x_ray_original_size is None:  ## Check for invalid input
            print("Could not open or find the image: {}. ".format(im_name_x_ray_original_size_test))
            shutil.copy2(im_name_x_ray_original_size, im_name_x_ray_original_size_test)
            print('Made a copy from {}\n'.format(im_name_x_ray_original_size))
            im_x_ray_original_size = cv2.imread(im_name_x_ray_original_size_test, cv2.IMREAD_GRAYSCALE)

        height, width = im_x_ray_original_size.shape[:2]  # height, width  -- original image size

        ratio = float(height) / width

        new_shape = (4 * 256, int(4 * 256 * ratio))

        im_x_ray_4x = cv2.resize(im_x_ray_original_size, new_shape)

        dir_img_x_ray_4x = 'results/bow-legs_test_4x/{}'.format(sub_dir_name)
        if not os.path.exists(dir_img_x_ray_4x):
            os.makedirs(dir_img_x_ray_4x)
        im_name_x_ray_4x = '{}/{}'.format(dir_img_x_ray_4x, file_name)
        cv2.imwrite(im_name_x_ray_4x, im_x_ray_4x)

        # mask
        im_mask_original_size = cv2.imread(file_name_in, cv2.IMREAD_GRAYSCALE)
        im_mask_4x = cv2.resize(im_mask_original_size, new_shape)
        im_name_mask_4x = '{}/{}'.format(dir_img_x_ray_4x, '/' + file_name_no_ext + '_mask_manual' + '.png')
        cv2.imwrite(im_name_mask_4x, im_mask_4x)

        # Unet output
        pr_openned_4x = cv2.resize(pr_openned, new_shape)
        print(filename_result)
        cv2.imwrite(filename_result, pr_openned_4x)

        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)

        with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
            print('{}  {:.4f} {:.4f}'.format(df.iloc[i][0], ious[i], dices[i]), file=f)

        i += 1
        if i == n_test:
            break

    print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean()))
    with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
        print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean()), file=f)
        print('\n', file=f)

    with open("results/bow-legs_IoU_Dice.txt", "a", newline="\r\n") as f:
        print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean()), file=f)