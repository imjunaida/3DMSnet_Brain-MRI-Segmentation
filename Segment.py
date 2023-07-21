# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 05:46:38 2019

@author: Junaid
"""
#setting Seed for proper training
import nibabel as nib
import numpy as np
seed = 7
np.random.seed(seed)

#Problem Configuration
num_classes = 3

patience = 1
model_filename = 'models/iSeg2017/outrun_step_{}.h5'
csv_filename = 'log/iSeg2017/outrun_step_{}.cvs'

nb_epoch = 20
validation_split = 0.25
class_mapper = {0 : 0, 10 : 0, 150 : 1, 250 : 2}
class_mapper_inv = {0 : 0, 1 : 10, 2 : 150, 3 : 250}


# General utils for reading and saving data
def get_filename(set_name, case_idx, input_name, loc='datasets') :
    pattern = '{0}/iSeg2017/iSeg-2017-{1}/subject-{2}-{3}.hdr'
    return pattern.format(loc, set_name, case_idx, input_name)

def get_set_name(case_idx) :
    return 'Training' if case_idx < 11 else 'Testing'

def read_data(case_idx, input_name, loc='datasets') :
    set_name = get_set_name(case_idx)

    image_path = get_filename(set_name, case_idx, input_name, loc)

    return nib.load(image_path)

def read_vol(case_idx, input_name, loc='datasets') :
    image_data = read_data(case_idx, input_name, loc)

    return image_data.get_data()[:, :, :, 0]

def save_vol(segmentation, case_idx, loc='results') :
    set_name = get_set_name(case_idx)
    input_image_data = read_data(case_idx, 'T1')

    segmentation_vol = np.empty(input_image_data.shape)
    segmentation_vol[:144, :192, :256, 0] = segmentation
    
    filename = get_filename(set_name, case_idx, 'label', loc)
    nib.save(nib.analyze.AnalyzeImage(
        segmentation_vol.astype('uint8'), input_image_data.affine), filename)

#N4BiasCorrection
import N4BiasRemoval as n4
for case_idx in range(18,24) :
    set_name = get_set_name(case_idx)
    image_path = get_filename(set_name, case_idx, 'T1', loc='datasets')
    n4.N4BiasCorrection(image_path)
    image_path = get_filename(set_name, case_idx, 'T2', loc='datasets')
    n4.N4BiasCorrection(image_path)
    image_path = get_filename(set_name, case_idx, 'label', loc='datasets')
    n4.N4BiasCorrection(image_path)



# Data preparation utils
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches

def extract_patches(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

def build_set(T1_vols, T2_vols, label_vols, extraction_step=(9, 9, 9)) :
    patch_shape = (27, 27, 27)
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 2, 27, 27, 27))
    y = np.zeros((0, 9 * 9 * 9, num_classes))
    for idx in range(len(T1_vols)) :
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), 2, 27, 27, 27))))
        y = np.vstack((y, np.zeros((len(label_patches), 9 * 9 * 9, num_classes))))

        for i in range(len(label_patches)) :
            y[i+y_length, :, :] = np_utils.to_categorical(label_patches[i].flatten(), num_classes)

        del label_patches

        # Sampling strategy: reject samples which labels are only zeros
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = T1_train[valid_idxs]
        del T1_train

        # Sampling strategy: reject samples which labels are only zeros
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step)
        x[y_length:, 1, :, :, :] = T2_train[valid_idxs]
        del T2_train
    return x, y

# Reconstruction utils
import itertools

def generate_indexes(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]

    idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]

    return itertools.product(*idxs)

def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

"""Architecture of the neural network"""
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Cropping3D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_model(num_classes) :
    init_input = Input((2, 27, 27, 27))

    x = Conv3D(25, kernel_size=(3, 3, 3))(init_input)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)

    y = Conv3D(50, kernel_size=(3, 3, 3))(x)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)

    z = Conv3D(75, kernel_size=(3, 3, 3))(y)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)

    x_crop = Cropping3D(cropping=((6, 6), (6, 6), (6, 6)))(x)
    y_crop = Cropping3D(cropping=((3, 3), (3, 3), (3, 3)))(y)

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = Conv3D(400, kernel_size=(1, 1, 1))(concat)
    fc = PReLU()(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)

    pred = Conv3D(num_classes, kernel_size=(1, 1, 1))(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, 9 * 9 * 9))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)

    model = Model(inputs=init_input, outputs=pred)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    return model


"""Segmenting initially"""
T1_vols = np.empty((10, 144, 192, 256))
T2_vols = np.empty((10, 144, 192, 256))
label_vols = np.empty((10, 144, 192, 256))
for case_idx in range(1, 11) :
    T1_vols[(case_idx - 1), :, :, :] = read_vol(case_idx, 'T1')
    T2_vols[(case_idx - 1), :, :, :] = read_vol(case_idx, 'T2')
    label_vols[(case_idx - 1), :, :, :] = read_vol(case_idx, 'label')

print(label_vols)
import matplotlib.pyplot as plt
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = T1_vols[:,10, :, :]
slice_1 = T2_vols[:, 10, :]
slice_2 = label_vols[:, :, 10]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
## Intensity normalisation (zero mean and unit variance)
T1_mean = T1_vols.mean()
T1_std = T1_vols.std()
T1_vols = (T1_vols - T1_mean) / T1_std
T2_mean = T2_vols.mean()
T2_std = T2_vols.std()
T2_vols = (T2_vols - T2_mean) / T2_std

# Combine labels of BG and CSF
for class_idx in class_mapper :
    label_vols[label_vols == class_idx] = class_mapper[class_idx]
    
"""Data Preparation"""    
x_train, y_train = build_set(T1_vols, T2_vols, label_vols, (3, 9, 3))

import nibabel as nib
epi_img = nib.load('p_label.nii.gz')
epi_img_data = epi_img.get_fdata()
epi_img_data.shape
 
