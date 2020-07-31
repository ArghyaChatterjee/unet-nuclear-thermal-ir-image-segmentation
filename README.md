# U-net-Deep-Convolutional-Neural-Network-for-Pool-boiling-IR-Image-segmentation
## Introduction:
- We evaluated qualitative as well as quantitative performance of different image processing and CNN based automatic methods with manually segmented results on Pool Boiling Infra-Red images by measuring important boiling heat transfer parameters statistically for each scheme.
- The time-lapsed high speed pool boiling IR image dataset used in this study unfortunately is not open source and one needs special permission to use this dataset as the dataset is produced inside expensive experimental setup of MIT’s Nuclear Reactor Laboratory.
## Demonstraion:
- Here is a short demonstration of my work. For better quality, see the demo on [youtube](https://www.youtube.com/watch?v=Px55bKnRF9A). 
<p align="center">
    <img src="Thesis.gif", width="800">
</p>

## Download Software:
- Download MIPAR software from the following link: https://www.mipar.us/
## Simple Operation:
- Load images inside MIPAR. 
- We first try a simple E-M thresholding based image processing technique to evaluate the segmentation performance. Image specific parameter tuning is essential as E-M thresholding is prone to detect background noise uniquely in each image. We use 8-pixel connectivity (face-based) for all images to determine local minima as the basis for the E-M threshold. 
- We set maximum difference of threshold pixel intensities to 38, 37, 25, 22, 40 and 26 against 600 kW/m^2 Heat Flux (HF) frame ids respectively; 26, 19, 23, 20, 28 and 29 against 700 kW/m^2 HF frame ids respectively; 24, 31, 28, 23, 34 and 21 against 800 kW/m^2 HF frame ids respectively; 38, 33, 30, 41, 35 and 58 against 900 kW/m^2 HF frame ids respectively and 38, 47, 36, 36, 43 and 40 against 1100 kW/m^2 HF frame ids respectively. 
- This parameter sets the maximum difference of threshold pixel intensities from the nearest local minima in order to be grouped into that minima's region.
## Complex Operation:
- A simple E-M threshold based segmentation method doesn’t provide accurate segmentation on a set of images. Therefore, several image processing steps are applied to obtain better segmentation performance. First, a contrast adjustment operation is performed on initial image to enhance dry spots relative to background. 
- A Non-Local Means Filtering operation is performed afterwards to make the background smooth. The pixel window size is set to 2 which is the size of neighborhood pixels considered about each pixel for the filter, and strength parameter is set to 0.096 which is a factor that controls the strength of the filter. 
- Then Adaptive Gaussian Thresholding operation is performed to initially segment the dry spots from their background. Again, the pixel window size is set to 6, select is set to dark and percentage value is set to 48.78% which means if a pixel is less than 48.78% of the average grayscale value in a 6 × 6 pixel window centered around it, it will be selected. 
- Morphological Erosion operation performed afterwards to remove background noise from binary segmented image. Threshold parameter is set to 5 which is the minimum number of empty pixels that must surround a selected pixel for it to be removed, and iterations is set to 10 which is the number of iterations to perform the erosion. However, erosion operation decreases the amount of dry areas. 
- Therefore, Morphological Dilation operation is performed to dilate the segmented dry spots in order to complement the loss of segmented areas in previous operation. The depth parameter is set to 3 which is the number of pixels by which selected regions will be grown in all directions.
## Special Thanks:
- Special thanks to my respectable supervisor Dr. Maglub Al Nur, Professor of Mechanical Engineering Department at Bangladesh University of Engineering and Technology (BUET), for providing me the opportunity to work on this topic and keeping faith in me that I can complete such a delicate task within this one year time period. His continuous support, motivation, friendly behaviour and pragmatic idea backed by helpful mentality really made this work as it is till now.
- Dr. Abir who is a Postdoctoral Research Associate at the Department of Nuclear Science and Engineering in Massachusetts Institute of Technology (MIT) and a co-supervisor of this work, provided constant support, guidance and resources during this work for which I am grateful to him. 
- Special thanks to Dr. Matteo Bucci, Professor of Nuclear Science and Engineering Department at Massachusetts Institute of Technology (MIT). He allowed me to use some of his lab data from their Expensive Setup inside Nuclear Reactor Laboratory for which I am grateful to him.
- Special thanks to MIPAR community for their support. They extended their trial version for me on my request. Their software was pretty helpful in all the image processing and post data processing operations inside my thesis.
## Citation:
- One who has access to the dataset can use the code to reproduce the result published in my undergrad thesis. You can find my undergrad thesis here: https://www.researchgate.net/publication/333371480
- If you want to cite the code, please add the doi number to the reference section of your research paper:
      [![DOI](https://zenodo.org/badge/224518388.svg)](https://zenodo.org/badge/latestdoi/224518388)
- If you want to cite this thesis, please mention it in the reference section of your paper like this way:
```
Chatterjee, Arghya. Computational Investigation on Pool Boiling IR Images for Segmentation of Dry Spots Automatically and Evaluating the Performance of Traditional Image Processing and Deep Neural Networks in Quantifying Dry Area Segments. Diss. Bangladesh University of Engineering and Technology, 2019.
```
- For further query regarding this work, you can contact me. My email address is: [arghyame20buet@gmail.com](mailto:arghyame20buet@gmail.com)

  
