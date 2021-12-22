# 2020.04.02
# @yifan

import numpy as np

from skimage.metrics import mean_squared_error

from core.util.bd_rate import BD_PSNR, BD_RATE
from core.util.ssim import structural_similarity, MultiScaleSSIM
    
def MSE(ref, X):
    if len(X.shape) == 4:
        mse = []
        for i in range(X.shape[0]):
            mse.append(mean_squared_error(ref[i], X[i]))
        return np.mean(mse)
    else:
        return mean_squared_error(ref, X)

def PSNR(ref, X, max_val=255):
    if len(X.shape) == 4:
        psnr = []
        for i in range(X.shape[0]):
            psnr.append(20*np.log10(max_val/np.sqrt(mean_squared_error(ref[i], X[i]))))
        return np.mean(psnr)
    else:
        return 20*np.log10(max_val/np.sqrt(mean_squared_error(XX, X)))

def BD(ref_R1, ref_PSNR1, R2, PSNR2):
    print('BD-PSNR: ', BD_PSNR(ref_R1, ref_PSNR1, R2, PSNR2))
    print('BD-RATE: ', BD_RATE(ref_R1, ref_PSNR1, R2, PSNR2))

def SSIM(ref, X, multichannel=True):
  if len(X.shape) == 4:
    ssim = []
    for i in range(X.shape[0]):
      ssim.append(structural_similarity(ref[i], X[i], multichannel=multichannel))
    return np.mean(ssim)
  return structural_similarity(ref, X, multichannel=multichannel)

def MS_SSIM(ref, X, max_val=255):
    return MultiScaleSSIM(ref, X, max_val=max_val)