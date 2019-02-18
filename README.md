# Realistic Noise Synthesis(sRGB)
## Introduction
This is the Python implementation of realistic noise synthesis on real images, followed by the approximated noise model proposed in CBDNet and Liu et al. The original matlab implementation verison is in [here](https://github.com/GuoShi28/CBDNet). This version is slightly different with the matlab one. We re-arrange the process order and adjust some incorrect details.   
## Realistic Noise Model (from CBDNet)
Given a clean image `x`, the realistic noise model can be represented as:
![](http://latex.codecogs.com/gif.latex?\\textbf{y}=f(M^{-1}(M(\\textbf{L}+n(\\textbf{x})))))
![](http://latex.codecogs.com/gif.latex?n(\\textbf{x})=n_s(\\textbf{x})+n_c)
Where `y` is the noisy image, `f(.)` is the CRF function which converts irradiance `L` to `x`. `M(.)` represents the function that convert sRGB image to Bayer image and `M^(-1)(.)` represents the demosaicing function.
If considering denosing on compressed images, 
![](http://latex.codecogs.com/gif.latex?\\textbf{y}=JPEG(f(M^{-1}(M(\\textbf{L}+n(\\textbf{x}))))))

## Usage
* "Test_Realistic_Noise_Model.py" is the testing script. Users can change the input image path, and adjust the \sigma_s, \sigma_c.

## Requirements and Dependencies
* Python 3


## Reference
[1.CBDNet](https://arxiv.org/abs/1807.04686)
[2.Liu et al. Automatic Estimation and Removal of Noise from a Single Image]https://ieeexplore.ieee.org/abstract/document/4359321
