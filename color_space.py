# 2020.01.28
# @yifan
# color space conversion
#
import numpy as np
import copy
import cv2
from sklearn.preprocessing import normalize
from skimage.measure import block_reduce

from myPCA import myPCA
from LLSR import LLSR

def Clip(X):
    tmp = copy.deepcopy(X)
    tmp = tmp.astype('int16')
    tmp[tmp > 255] = 255
    tmp[tmp < 0] = 0
    return tmp
    
def BGR2PQR(X, doClip=True, doRescale=False):
    def reScale(K):
        K[:1] = normalize(K[:1], norm='l1')
        K[0] *= 219/255
        sb = 224/255/(np.sum(np.abs(K[1])))
        K[1] *= sb
        sc = 224/255/(np.sum(np.abs(K[2])))
        K[2] *= sc
        return K
    pca = myPCA()
    S = X.shape
    X = X.reshape(-1, 3)
    pca.fit(X)
    if doRescale == True:
        pca.Kernels = reScale(pca.Kernels)
    tX = pca.transform(X).reshape(S)
    if doClip == True:
        tX = np.round(tX)
    def sum_neg(K):
        s = 0
        for i in range(len(K)):
            if K[i] < 0:
                s -= K[i]
        return s
    if np.abs(np.min(tX)) > np.abs(np.max(tX)):
        pca.Kernels *= -1
        tX *= -1
    offset = [16, sum_neg(pca.Kernels[1])*255+16, sum_neg(pca.Kernels[2])*255+16]
    #offset = [0,0,0]
    tX += np.array(offset)  
    return (pca, offset), tX

def PQR2BGR(X, pca, doClip=False):
    X -= pca[1]
    if doClip == True:
        return Clip(pca[0].inverse_transform(X))
    return pca[0].inverse_transform(X)

def BGR2RGB(X):
    R        = copy.deepcopy(X[:,:,2:])
    G        = copy.deepcopy(X[:,:,1:2])
    B        = copy.deepcopy(X[:,:,0:1])
    return np.concatenate((R, G, B), axis=-1)

def RGB2BGR(X):
    B        = copy.deepcopy(X[:,:,2:])
    G        = copy.deepcopy(X[:,:,1:2])
    R        = copy.deepcopy(X[:,:,0:1])
    return np.concatenate((B, G, R), axis=-1)

def RGB2YUV(X, doClip=False):
    K = np.array([[   0.299,    0.587,    0.114],
                  [0.596, 0.274 ,    - 0.322],
                  [  0.211   , 0.523   , 0.312]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3,-1))
    X = np.moveaxis(X.reshape(S), 0, -1)
    if doClip == True:
        return np.round(X)
    return X

def BGR2YUV(X, doClip=False):
    return RGB2YUV(BGR2RGB(X), doClip=doClip)
    
def YUV2RGB(X, doClip=False):
    K = np.array([[1,        0.956,  0.621],
                  [1, - 0.272 , - 0.647],
                  [1,  - 1.106,       1.703]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3,-1))
    X = np.moveaxis(X.reshape(S), 0, -1)
    if doClip == True:
        return Clip(X)
    return X

def YUV2BGR(X, doClip=False):
    return RGB2BGR(YUV2RGB(X))

def c444_to_c420(YUV):
    return [YUV[:,:,0], block_reduce(YUV[:,:,1], (2,2), np.mean), block_reduce(YUV[:,:,2], (2,2), np.mean)]

def c420_to_c444(X):
    S = X[0].shape
    tmp = [ X[0].reshape(S[0],S[1],1), cv2.resize(X[1], (S[1], S[0])).reshape(S[0],S[1],1), cv2.resize(X[2], (S[1], S[0])).reshape(S[0],S[1],1) ]
    return np.concatenate(tmp, axis=-1)

def ML_inv_color(X_bgr, iX, doClip=True):
    llsr = LLSR(onehot=False)
    llsr.fit(iX.reshape(-1,3), X_bgr.reshape(-1,3))
    iX = llsr.predict_proba(iX.reshape(-1,3)).reshape(X_bgr.shape)
    if doClip == True:
        return Clip(iX)
    return iX
