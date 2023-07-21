# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 08:25:31 2019

@author: Junaid
"""
import nibabel as nib
import SimpleITK as sitk
def N4BiasCorrection(input_path):
    inputImage = sitk.ReadImage(input_path)
# maskImage = sitk.ReadImage("06-t1c_mask.nii.gz")
    maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
    #maskImagePath=input_path.replace('Testing','N4Test')
    maskImagePath=input_path.replace('.nii.gz','-mask.nii.gz')
    sitk.WriteImage(maskImage, maskImagePath)
    print("Mask image is saved.")
    
    
    inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    
    output = corrector.Execute(inputImage,maskImage)
    outputPath = maskImagePath.replace('-mask','-bias')
    sitk.WriteImage(output,outputPath)
    print("Finished N4 Bias Field Correction.....")

N4BiasCorrection('brainweb.nii.gz')