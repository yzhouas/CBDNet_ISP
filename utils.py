'''
CBDNet Code Reimplementation
By Yuqian Zhou @ Megvii USA
'''
import numpy as np
import cv2
import os
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  #scipy fit curve

#from colour_demosaicing import (
#    EXAMPLES_RESOURCES_DIRECTORY,
#    demosaicing_CFA_Bayer_bilinear,
#    demosaicing_CFA_Bayer_Malvar2004,
#    demosaicing_CFA_Bayer_Menon2007,
#    mosaicing_CFA_Bayer)

#gamma function for approximation the CRF function
def func(x, a):
    return np.power(x, a)

#fit the curve using the pre-defined function and return the fit parameter
def CRF_curve_fit(I, B):
    popt, pcov = curve_fit(func, I, B)
    #print(popt)   
    return popt

#CRF: I-->B  iCRF: B-->I
def CRF_function_transfer(x, y):
    para = []
    for crf in range(201):
        temp_x = np.array(x[crf, :])
        temp_y = np.array(y[crf, :])
        para.append(CRF_curve_fit(temp_x, temp_y))
    return para  #save all the parameters for fitting

def mosaic_bayer(rgb, pattern, noiselevel):
    '''
    Mosaic Function
    [Input] rgb: full rgb image
            pattern: mosaic pattern #1: grbg 2:rggb 3:bgrg 4:bggr 5: no mosaic
            noiselevel : noise level of gaussian noise
    pattern = 'grbg' 
            G R
            B G
    pattern = 'rggb'
            R G
            G B
    pattern = 'gbrg'
            G B
            R G
    pattern = 'bggr'
            B G
            G R
    [Output] mosaic: mosaiced image
             
    '''
    w, h, c = rgb.shape
    if pattern == 1: #gbrg
        num = [1, 2, 0, 1]
    elif pattern == 2:  #grbg
        num = [1, 0, 2, 1]
    elif pattern == 3:  #bggr
        num = [2, 1, 1, 0]
    elif pattern == 4:  #rggb
        num = [0, 1, 1, 2]
    elif pattern == 5:
        return rgb  #no mosaic
    
    mosaic = np.zeros((w, h, 3))
    mask = np.zeros((w, h, 3))
    B = np.zeros((w, h))

    B[0:w:2, 0:h:2] = rgb[0:w:2, 0:h:2, num[0]]
    B[0:w:2, 1:h:2] = rgb[0:w:2, 1:h:2, num[1]]
    B[1:w:2, 0:h:2] = rgb[1:w:2, 0:h:2, num[2]]
    B[1:w:2, 1:h:2] = rgb[1:w:2, 1:h:2, num[3]]

    #add gaussian noise
    gauss = np.random.normal(0, noiselevel/255.,(w, h))
    gauss = gauss.reshape(w, h)
    B = B + gauss

    #compute mask
#    mask[0:w:2, 0:h:2, num[0]] = 1
#    mask[0:w:2, 1:h:2, num[1]] = 1
#    mask[1:w:2, 0:h:2, num[2]] = 1
#    mask[1:w:2, 1:h:2, num[3]] = 1
#
#    #compute mosaic
#    mosaic[0:w:2, 0:h:2, num[0]] = B[0:w:2, 0:h:2]
#    mosaic[0:w:2, 1:h:2, num[1]] = B[0:w:2, 1:h:2]
#    mosaic[1:w:2, 0:h:2, num[2]] = B[1:w:2, 0:h:2]
#    mosaic[1:w:2, 1:h:2, num[3]] = B[1:w:2, 1:h:2]

    return (B, mask, mosaic)

def ICRF_Map(Img, I, B):
    '''
    Transfer Irradiance L to image x according to CRF function
    [Input]: Img: input irradiance L, single or double image type, [0,1]
             I, B: inverse CRF response lookup table
    [Output]:
             Img: image x
    '''
    w, h, c = Img.shape
    #print(Img.shape)
    output_Img = Img.copy()
    #print(output_Img.shape)
    prebin = I.shape[0]
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    for i in range(w):
       for j in range(h):
           for k in range(c):  #go over all the pixels
               temp = output_Img[i, j, k]
               start_bin = 0
               if temp > min_tiny_bin:
                   start_bin = math.floor(temp/tiny_bin - 1) - 1
               for b in range(start_bin, prebin):  #check the range save as the matlab code!
                   tempB = B[b]
                   if tempB >= temp:
                       index = b
                       if index > 0:
                           comp1 = tempB - temp
                           comp2 = temp - B[index-1]
                           if comp2 < comp1:
                               index = index - 1
                       output_Img[i, j, k] = I[index]
                       break
               
    return output_Img

def CRF_Map(Img, I, B):
    '''
    Transfer image x to irradiance L accroding to CRF function
    [Input]
    Img: Input np array [0,1] double
    I, B: CRF response lookup table
    [output] Irradiance L
    '''
    w, h, c = Img.shape
    output_Img = Img.copy()
    prebin = I.shape[0]
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    for i in range(w):
        for j in range(h):
            for k in range(c):
                temp = output_Img[i, j, k]
                #clipping operation
                if temp < 0:
                    temp = 0
                    Img[i, j, k] = 0
                elif temp > 1:
                    temp = 1
                    Img[i, j, k] = 1
                start_bin = 0
                if temp > min_tiny_bin:
                    start_bin = math.floor(temp/tiny_bin - 1) - 1
                for b in range(start_bin, prebin): #check if consistent with matlab
                    tempB = I[b]
                    if tempB >= temp:
                        index = b
                        if index > 0:
                            comp1 = tempB - temp
                            comp2 = temp - B[index-1]
                            if comp2 < comp1:
                                index = index - 1
                        output_Img[i, j, k] = B[index]
                        break
    return output_Img 

def CRF_Map_opt(Img, popt):
    '''
    Optimized Version of CRF mapping function
    Transfer image x to irradiance L accroding to CRF function
    [Input]
    Img: Input np array [0,1] double
    I, B: CRF response lookup table
    [output] Irradiance L
    '''
    w, h, c = Img.shape
    output_Img = Img.copy()
    #print(popt)
    output_Img = func(output_Img, *popt)  #directly output according to the image
    return output_Img 

def AddMCAWGN(x, sigma_c):
    '''
    Add MC-AWGN to the image
    '''
    w, h, c = x.shape
    temp_x = x.copy()
    noise_c = np.zeros((w, h, c))
    for chn in range(3):
        noise_c [:, :, chn] = np.random.normal(0, sigma_c[chn], (w, h))
    temp_x = temp_x + noise_c
    cv2.imwrite('AWGN.png', temp_x * 255.)
    return temp_x

def Demosaic(B_b, pattern):
     
    B_b = B_b * 255
    B_b = B_b.astype(np.uint16)

    #print(B_b)
    if pattern == 1:
        #lin_rgb = colour_demosaicing.bayer.demosaicing.malvar2004(B_b)
        #lin_rgb = demosaicing_CFA_Bayer_Malvar2004(B_b) 
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerGB2BGR)
    elif pattern == 2:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerGR2BGR)
    elif pattern == 3:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerBG2BGR)
    elif pattern == 4:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerRG2BGR)
    elif pattern == 5:
        lin_rgb = B_b

    lin_rgb = lin_rgb[:,:,::-1] / 255.  
    return lin_rgb

def AddNoiseMosai(x, CRF_para, iCRF_para, I, B, Iinv, Binv, sigma_s, sigma_c, crf_index, pattern, opt = 1):
    '''
    Add Realistic Noise according to the pre-defined model
    I = M^{-1}(M(f(L + n_s + n_c) + n_q))
    n_s: shot noise, depend on L, E[n_s]=0, var[n_s] = L*sigma_s
            in [1], n_s is a Possion Shot
            in [2], n_s is GMM, sigma_s: 0-0.16
            [2] is chosen here
    n_c: other type of noise, i.e. read noise, dark current
            in [2], E[n_c]=0, var[n_c]=sigma_c, sigma_c:0.01-0.06
    n_q: ignore in [2]
    [Inputs]
    x: np array Image data double (0,1), w*h*c
    I, B: CRF parameters provided by [3]
    Iinv, Binv: inverse CRF parameters, created by "InverseCRF" for faster computation
    (optional) If not defined by user, the following parameters are set randomly
    sigma_s: signal_dependent noise level [0, 0.16] and it's channel-dep
    sigma_c: signal_independent noise level, [0, 0.06] and it's channel-dep
    crf_index: CRF_index, [1, 201]
    pattern: 1: 'gbrg' 2:'grbg' 3: 'bggr' 4: 'rggb' 5: no mosaic
    opt: [0] series [1] parallel and difference
    [Output]
    y: noisy_image
    [Reference]
    [1] G.E. Healey and R. Kondepudy, Radiometric CCD Camera Calibration and Noise Estimation,
     IEEE Trans. Pattern Analysis and Machine Intelligence
    [2] Liu, Ce et al. Automatic Estimation and Removal of Noise from a Single Image. 
     IEEE Transactions on Pattern Analysis and Machine Intelligence 30 (2008): 299-314.
    [3] Grossberg, M.D., Nayar, S.K.: Modeling the space of camera response functions.
     IEEE Transactions on Pattern Analysis and Machine Intelligence 26 (2004)
    '''
    w, h, c = x.shape
    #cv2.imwrite('original.png', (x * 255).astype('int'))
    #x --> L: clean image to irraidence for matching
    #temp_x = ICRF_Map(x, Iinv[crf_index, :], Binv[crf_index, :])
    temp_x = CRF_Map_opt(x, iCRF_para[crf_index] )
    #print(temp_x.shape)
    #cv2.imwrite('1_x2L.png', (temp_x * 255).astype('int'))
    #add noise related to the clean image and add it to L
    #add signal_dependent noise to L
    sigma_s = np.reshape(sigma_s, (1, 1, c))  #reshape the sigma factor to [1,1,c] to multiply with the image
    noise_s_map = np.multiply(sigma_s, temp_x)  #according to temp_x
    #print(noise_s_map)           # different from the official code, here we use the original clean image x to compute the variance
    noise_s = np.random.randn(w, h, c) * noise_s_map  #use the new variance to shift the normal distribution
    temp_x_n = temp_x + noise_s
    #cv2.imwrite('2_L_noises.png', temp_x * 255.)
    #add signal_independent noise to L 
    noise_c = np.zeros((w, h, c))
    for chn in range(3):
        noise_c [:, :, chn] = np.random.normal(0, sigma_c[chn], (w, h)) 
    temp_x_n = temp_x_n + noise_c
    #clipping process
    temp_x_n = np.clip(temp_x_n, 0.0, 1.0)
    #cv2.imwrite('3_L_noisec.png', temp_x * 255.)
    #L --> x
    #temp_x = CRF_Map(temp_x, I[crf_index, :], B[crf_index, :])
    temp_x_n = CRF_Map_opt(temp_x_n, CRF_para[crf_index])
    if opt == 1:
        temp_x = CRF_Map_opt(temp_x, CRF_para[crf_index])
    #cv2.imwrite('4_L2x.png', temp_x * 255.)
    #add Mosaic
    
    B_b_n = mosaic_bayer(temp_x_n[:,:,::-1], pattern, 0)[0]  #here pattern can be 1, 2, 3, 4, 5
    #cv2.imwrite('5_add_mosaic.png', B_b * 255)
    lin_rgb_n = Demosaic(B_b_n, pattern)
    result = lin_rgb_n 
    #cv2.imwrite('noise_demosaic.png', result[:,:,::-1] * 255)
    if opt == 1:
        B_b = mosaic_bayer(temp_x[:,:,::-1], pattern, 0)[0]  #here pattern can be 1, 2, 3, 4, 5
    #cv2.imwrite('5_add_mosaic.png', B_b * 255)
        lin_rgb = Demosaic(B_b, pattern)
        #cv2.imwrite('original_demosaic.png', lin_rgb[:,:,::-1] * 255)
        diff = lin_rgb_n - lin_rgb
        #cv2.imwrite('diff.png', diff[:,:,::-1] * 255)
        result = x + diff
    
    #print(temp_x)
    #cv2.imwrite('6_demosaic.png', lin_rgb * 255.)

    return result

def generate_real_noise(image, I_gl, B_gl, I_inv_gl, B_inv_gl, mode=1):
    '''
    Generate noisy images according to the above noise model
    [Input] Image: [0,1]double 
            mode: uncompressed noisy image
    '''
    sigma_s = np.random.uniform(0.0, 0.16, (3,))
    sigma_c = np.random.uniform(0.0, 0.06, (3,))
    CRF_index = np.random.choice(201)
    pattern = np.random.choice(4) + 1  #must use the mosaic to convert and back
    noisy = AddNoiseMosai(image, I_gl, B_gl, I_inv_gl, B_inv_gl, sigma_s, sigma_c, CRF_index, pattern)
    
    return noisy

def generate_GP_noise(image, sigma_s, sigma_c):
    '''
    Generate Gaussian-possion noise: 
    n(x) = x + ns + nc, \sigma(ns) = x * \sigma_s , \sigma(nc) = \sigma_c 
    ns is signal-dependent
    nc is signal-independent AWGN
    [Input] Image with [0,1]
    [Output] Noisy image with randomly sampled noise according to the noise model
    '''
    w, h, c = image.shape
    temp_x = image
    #sigma_s = np.random.uniform(0.0, 0.16, (3,))
    #sigma_c = np.random.uniform(0.0, 0.06, (3,))
    
    sigma_s = np.reshape(sigma_s, (1, 1, c))  #reshape the sigma factor to [1,1,c] to multiply with the image
    noise_s_map = np.multiply(sigma_s, image)  #according to x or temp_x?? (according to clean image or irradience)
    #print(noise_s_map)           # different from the official code, here we use the original clean image x to compute the variance
    noise_s = np.random.randn(w, h, c) * noise_s_map  #use the new variance to shift the normal distribution
    temp_x = temp_x + noise_s
    #add signal_independent noise to L 
    noise_c = np.zeros((w, h, c))
    for chn in range(3):
        noise_c [:, :, chn] = np.random.normal(0, sigma_c[chn], (w, h))
    temp_x = temp_x + noise_c

    return temp_x

def generate_ms_GP_noise(image, scale=1.0):
    '''
    Generate Multi-scale Gaussian-possion noise: 
    Downscale the image to generate noise, and upsample the noise map
    add it back to the original size of image
    n(x) = x + ns + nc, \sigma(ns) = x * \sigma_s , \sigma(nc) = \sigma_c 
    ns is signal-dependent
    nc is signal-independent AWGN
    [Input] Image with [0,1]
    [Output] Noisy image with randomly sampled noise according to the noise model
    '''
    w, h, c = image.shape
    temp_x = image.copy()
    temp_x_resize = cv2.resize(temp_x, (0,0), fx=scale, fy=scale, interpolation= cv2.INTER_AREA)
    w2, h2, c2 = temp_x_resize.shape  #new size
    sigma_s = np.random.uniform(0.0, 0.16, (3,))
    #sigma_s = [0.13,0.13,0.13]
    sigma_c = np.random.uniform(0.0, 0.06, (3,))
    #sigma_c = [0.01,0.01,0.01]
    sigma_s = np.reshape(sigma_s, (1, 1, c))  #reshape the sigma factor to [1,1,c] to multiply with the image
    noise_s_map = np.multiply(sigma_s, temp_x_resize)  #according to x or temp_x?? (according to clean image or irradience)
    #print(noise_s_map)           # different from the official code, here we use the original clean image x to compute the variance
    noise_s = np.random.randn(w2, h2, c2) * noise_s_map  #use the new variance to shift the normal distribution
    temp_x = temp_x + cv2.resize(noise_s, dsize=(h, w), interpolation = cv2.INTER_AREA)
    #add signal_independent noise to L 
    noise_c = np.zeros((w2, h2, c2))
    for chn in range(3):
        noise_c [:, :, chn] = np.random.normal(0, sigma_c[chn], (w2, h2))
    temp_x = temp_x + cv2.resize(noise_c, dsize=(h, w), interpolation = cv2.INTER_AREA)
    return temp_x

