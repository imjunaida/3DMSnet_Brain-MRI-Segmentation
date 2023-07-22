# 3DSMSnet- Brian MRI Segmentation
 This method aims at segmenting Brain MRI into Grey matter (GM), White matter(WM), and Cerebrospinal Fluid (CSF). This work was done as part of fun capstone project during my undergraduate program at SRM Institute of Science and Technology(Chennai).
 Here we make use of 3D CNN to do a patch-based segmentation of the slices of the MRI. The MRI images are first pre-processed which involves bias field correction and splitting the MRI into optimal patch size of 25x25x25.

 This method is inspired by the work of *Jose Dolz et al.*: [3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study](https://www.sciencedirect.com/science/article/abs/pii/S1053811917303324)

# Pipeline of the system

 ![Pipeline of System](https://github.com/imjunaida/3DMSnet_Brain-MRI-Segmentation/blob/master/Plots/mri3.png?raw=true)

# Running the trained models

The model was trained on a mixture of iSeg2017 data and synthetic data generated by a group at McGill University [(BrainWeb)](https://brainweb.bic.mni.mcgill.ca/)
<br>**1.** Set the path of file in *segment.py* to the your dataset directory
<br>**2.** In the for loop define the number of test files you are running
<br>**3.** Run the *testing.py* file and the results will be saved in a newly created **results** directory

# Result 
<p align="center">
    <img src="https://github.com/imjunaida/3DMSnet_Brain-MRI-Segmentation/blob/master/Plots/Ground-Truth.gif" width= 250 height= auto>
    
   <img src="https://github.com/imjunaida/3DMSnet_Brain-MRI-Segmentation/blob/master/Plots/our_result.gif" width= 250 height= auto>
  
</p>
<p align ="center"> <em>Ground Truth vs Segmented Results</em></p>

