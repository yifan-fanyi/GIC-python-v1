# 2020.04.02
# @yifan

import numpy as np
import time
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

from bd_rate import BD_PSNR, BD_RATE
from ssim import structural_similarity, MultiScaleSSIM

def Time(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("   <RunTime> %s: %4.1f s"%(method.__name__, (te - ts)))
        return result
    return timed
    
def MSE(ref, X):
    return mean_squared_error(ref, X)

def PSNR(X, XX):
    return 20*np.log10(255/np.sqrt(mean_squared_error(XX, X)))

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