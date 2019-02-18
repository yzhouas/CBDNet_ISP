# Realistic_Noise_Synthesis_sRGB
(https://arxiv.org/abs/1807.04686)
## Introduction
This is the Python implementation of realistic noise synthesis on real images. 
## Realistic Noise Model
Given a clean image `x`, the realistic noise model can be represented as:

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=f(M^{-1}(M(\\textbf{L}+n(\\textbf{x})))))

![](http://latex.codecogs.com/gif.latex?n(\\textbf{x})=n_s(\\textbf{x})+n_c)

Where `y` is the noisy image, `f(.)` is the CRF function which converts irradiance `L` to `x`. `M(.)` represents the function that convert sRGB image to Bayer image and `M^(-1)(.)` represents the demosaicing function.

If considering denosing on compressed images, 

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=JPEG(f(M^{-1}(M(\\textbf{L}+n(\\textbf{x}))))))

## Usage
* "Test_Patches.m" is the testing code for small images or image patches. If the tesing image is too large (e.g., 5760*3840), we recommend to use "Test_fullImage.m"
*  "Test_fullImage.m" is the testing code for large images. 
*  "Test_Realistic_Noise_Model.m" is the testing code for the realistic noise mode in our paper. And it's very convinent to utilize [AddNoiseMosai.m](https://github.com/GuoShi28/CBDNet/blob/master/utils/AddNoiseMosai.m) to train your own denoising model for real photographs.

## Requirements and Dependencies
* Python 3


## Citation
[https://arxiv.org/abs/1807.04686](https://arxiv.org/abs/1807.04686)

