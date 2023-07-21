# 3DSMSnet- Brian MRI Segmentation
 This method aims at segmenting Brain MRI into Grey matter (GM), White matter(WM), and Cerebrospinal Fluid (CSF). This work was done as part of fun capstone project during my undergraduate program at SRM Institute of Science and Technology(Chennai).
 Here we make use of 3D CNN to do a patch-based segmentation of the slices of the MRI. The MRI images are first pre-processed which involves bias field correction and splitting the MRI into optimal patch size of 25x25x25.

 This method is inspired by the work of *Jose Dolz et al.*: [3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study](https://www.sciencedirect.com/science/article/abs/pii/S1053811917303324)

 # Pipeline of the system

 ![alt text](https://github.com/imjunaida/3DMSnet_Brain-MRI-Segmentation/blob/master/Plots/mri3.png?raw=true)

