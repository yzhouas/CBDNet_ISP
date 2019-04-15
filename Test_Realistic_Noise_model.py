'''
%% Test Code for realistic noise model from 
%%% https://arxiv.org/abs/1807.04686
Implemented by Yuqian Zhou @Megvii USA
'''

import numpy as np
import cv2
import scipy.io as sio
import os
from utils import *

'''
Realistic Noise Model Introduced in CBDNet
y = M^{-1}(M(f(L + n(x)))), L = f^{-1}(x) 
x and y are the original clean image and the noisy image we created. 
n(x) = n_s(x) + n_c, 
Var(n_s(x)) = \sigma_s * x, Var(n_c) = \sigma_c

'''
CRF = sio.loadmat('201_CRF_data.mat')
iCRF = sio.loadmat('dorfCurvesInv.mat')
B_gl = CRF['B']
I_gl = CRF['I']
B_inv_gl = iCRF['invB']
I_inv_gl = iCRF['invI']

if os.path.exists('201_CRF_iCRF_function.mat')==0:
    #fitting CRF function hereddNoiseMosa
    CRF_para = np.array(CRF_function_transfer(I_gl, B_gl))
    #fitting iCRF function here
    #iCRF_para = CRF_function_transfer(B_inv_gl, I_inv_gl)
    iCRF_para = 1. / CRF_para
    sio.savemat('201_CRF_iCRF_function.mat', {'CRF':CRF_para, 'iCRF':iCRF_para})
else:
    Bundle = sio.loadmat('201_CRF_iCRF_function.mat')
    CRF_para = Bundle['CRF']
    iCRF_para = Bundle['iCRF']

path = '1.jpg'
Img = cv2.imread(path)
Img = Img[:,:,::-1]  #Change it to RGB channel order
Img = (Img/255.) #convert it to RGB channel order
Img = np.array(Img).astype('float32')
#print(Img)
#model 1: randomly choose \sigma_s, \sigma)c, CRF and mosaic pattern
#noise = AddNoiseMosai(Img, I_gl, B_gl, I_inv_gl, B_inv_gl)

#model 2: pre-defined setting \sigma_s, \sigma_c, CRF and mosaic pattern
sigma_s = [0.0, 0.0, 0.0]  #0-0.16
sigma_c = [0.005, 0.005, 0.005]  #0-0.06

CRF_index = 4  #1-201
pattern = 1  #1: gbrg 2:grbg 3:bggr 4:rggb 5: no mosaic
noise = AddNoiseMosai(Img, CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl, sigma_s, sigma_c, CRF_index, pattern, 0)  #mode 0: without difference
noise1 = AddNoiseMosai(Img, CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl, sigma_s, sigma_c, CRF_index, pattern, 1)  #mode 1: with difference

awgn = AddMCAWGN(Img, sigma_c) 
gp = generate_GP_noise(Img, sigma_s, sigma_c)
cv2.imwrite('GP.png', gp[:,:,::-1]*255)
#print(noise)
#JPEG compression
quality = 70  #set 100 to get full quality if JPEG is not considered
path_temp = os.path.join('./', 'jpeg.jpg')
path_temp1 = os.path.join('./', 'jpeg1.jpg')
#write image with JPEG qualtiy compressin
cv2.imwrite(path_temp, noise[:,:,::-1] * 255,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
cv2.imwrite(path_temp1, noise[:,:,::-1] * 255, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

#load the compressed jpeg image back
output = cv2.imread(path_temp)



