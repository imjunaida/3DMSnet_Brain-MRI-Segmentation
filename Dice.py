# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 05:16:37 2019

@author: Junaid
"""

import nibabel as nib
import numpy as np
dice=0
for i in range(1,6):
    img1 =nib.load('datasets/iSeg2017/iSeg-2017-Training/subject-{}-label.hdr'.format(i))
    data1 = img1.get_fdata()
  
    gt = np.rot90(data1.squeeze(), 1)

    img =nib.load('results/iSeg2017/iSeg-2017-Training/subject-{}-label.hdr'.format(i))
    data = img.get_fdata()
    
    seg = np.rot90(data.squeeze(), 1)

    dice += np.sum(seg[gt==250]==250)*2.0 / (np.sum(seg[seg==250]==250) + np.sum(gt[gt==250]==250))
print(dice/5)