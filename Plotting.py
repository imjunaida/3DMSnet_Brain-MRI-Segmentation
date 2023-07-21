# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 03:45:47 2019

@author: Junaid
"""
import SimpleITK as sitk
import numpy as np
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing
import nibabel as nib
import numpy as np
img =nib.load('subject-14-label.hdr')
data = img.get_fdata()
print(data.shape)
data = np.rot90(data.squeeze(), 1)
import matplotlib.pyplot as plt
%matplotlib qt
fig, ax = plt.subplots(1, 6, figsize=[30, 3])
n = 0
slice = 110
for _ in range(6):
    ax[n].imshow(data[:,:,slice],'gray')
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    ax[n].set_title('Slice number: {}'.format(slice), color='r')
    n += 1
    slice += 5
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()
