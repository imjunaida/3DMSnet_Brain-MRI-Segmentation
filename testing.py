# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 06:58:59 2019

@author: Junaid
"""
from keras.models import load_model

# Load best model
model = generate_model(num_classes)
model.load_weights(model_filename.format(1))

for case_idx in range(11, 13) :
    T1_test_vol = read_vol(case_idx, 'T1')[:144, :192, :256]
    T2_test_vol = read_vol(case_idx, 'T2')[:144, :192, :256]
    
    x_test = np.zeros((6916, 2, 27, 27, 27))
    x_test[:, 0, :, :, :] = extract_patches(T1_test_vol, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
    x_test[:, 1, :, :, :] = extract_patches(T2_test_vol, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
    
    x_test[:, 0, :, :, :] = (x_test[:, 0, :, :, :] - T1_mean) / T1_std
    x_test[:, 1, :, :, :] = (x_test[:, 1, :, :, :] - T2_mean) / T2_std

    pred = model.predict(x_test, verbose=2)
    pred_classes = np.argmax(pred, axis=2)
    pred_classes = pred_classes.reshape((len(pred_classes), 9, 9, 9))
    segmentation = reconstruct_volume(pred_classes, (144, 192, 256))
    
    csf = np.logical_and(segmentation == 0, T1_test_vol != 0)
    segmentation[segmentation == 2] = 250
    segmentation[segmentation == 1] = 150
    segmentation[csf] = 10
    
    save_vol(segmentation, case_idx)
    
    print("Finished segmentation of case # {}".format(case_idx))
