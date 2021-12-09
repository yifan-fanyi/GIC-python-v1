import numpy as np
import os
import pickle
import copy 
import json
import warnings
warnings.filterwarnings("ignore")
from load_img import Load_from_Folder, Load_Images
from evaluate import MSE, PSNR
from util import *
from LANCZOS import *
from GIC import GIC

def downsample(X, r):
        a = []
        for i in range(0, X.shape[0], 500):
            aa, _ =  LANCZOS.split(X[i:i+500], r)  
            a.append(aa)
        return np.concatenate(a, axis=0)

def run1():
    par_old1 = {'n_hop': 6, 
           'win': {'hop1': [256, 128, 64, 32, 16, 8], 
                   'hop2': [128, 64, 32, 16, 8, 4], 
                   'hop3': [64, 32, 16, 8, 4], 
                   'hop4': [32, 16, 8, 4], 
                   'hop5': [16, 8, 4], 
                   'hop6': [8, 4]}, 
            'n_clusters': {'hop1': [64, 128, 512, 512, 512, 64], 
                    'hop2': [64, 128, 512, 512, 512, 128], 
                    'hop3': [64, 128, 512, 512, 512], 
                    'hop4': [64, 128, 512, 512], 
                    'hop5': [64, 32, 64], 
                    'hop6': [64, 32]},   
           'n_subcodebook': {'hop1': [1, 1, 1, 1, 1, 1], 
                             'hop2': [1, 1, 1, 1, 1, 1], 
                             'hop3': [1, 1, 1, 1, 1], 
                             'hop4': [1, 1, 1, 1], 
                             'hop5': [1, 1, 1], 
                             'hop6': [1, 1]}, 
           'shape': {'hop1': [[-1, 1, 1, 1], [-1, 2, 2, 1], [-1, 4, 4, 1], [-1, 8, 8, 1], [-1, 16, 16, 1], [-1, 32, 32, 1]], 
                     'hop2': [[-1, 1, 1, 1], [-1, 2, 2, 1], [-1, 4, 4, 1], [-1, 8, 8, 1], [-1, 16, 16, 1], [-1, 32, 32, 1]], 
                     'hop3': [[-1, 1, 1, 1], [-1, 2, 2, 1], [-1, 4, 4, 1], [-1, 8, 8, 1], [-1, 16, 16, 1]], 
                     'hop4': [[-1, 1, 1, 1], [-1, 2, 2, 1], [-1, 4, 4, 1], [-1, 8, 8, 1]], 
                     'hop5': [[-1, 1, 1, 1], [-1, 2, 2, 1], [-1, 4, 4, 1]], 
                     'hop6': [[-1, 1, 1, 1], [-1, 2, 2, 1]]}, 
           'DC_n_components': {'hop1': [150, 150, 150, 150, 150, None], 
                               'hop2': [150, 150, 150, 150, 150, None], 
                               'hop3': [150, 150, 150, 150, None], 
                               'hop4': [150, 150, 150, None], 
                               'hop5': [150, 150, None], 
                               'hop6': [150, None]}, 
           'AC_n_components': {'hop1': [None, None, None, None, None, None], 
                               'hop2': [None, None, None, None, None, None], 
                               'hop3': [None, None, None, None, None], 
                               'hop4': [None, None, None, None], 
                               'hop5': [None, None, None], 
                               'hop6': [None, None]}, 
            'n_components': {'hop1': [None, None, None, None, None, None], 
                               'hop2': [None, None, None, None, None, None], 
                               'hop3': [None, None, None, None, None], 
                               'hop4': [None, None, None, None], 
                               'hop5': [None, None, None], 
                               'hop6': [None, None]}, 
           'MSE_TH': {'hop1': [70, 70, 70, 70, 70, 70], 
                      'hop2': [70, 70, 70, 70, 70, 70], 
                      'hop3': [70, 70, 70, 70, 70], 
                      'hop4': [70, 70, 70, 70], 
                      'hop5': [70, 70, 70], 
                      'hop6': [70, 70]}, 
           'n_threads': 12,
           'shared_CB':   { 'MSE_TH_4':70, 
                            'MSE_TH_8':70, 
                            'MSE_TH_16':70, 
                            'MSE_TH_32':70, 
                            'MSE_TH_64':70, 
                            'MSE_TH_128':70,
                            'MSE_TH_256':70, 
                            'MSE_TH_512':70, 
                            'MSE_TH_1024':70,  
                            'load_4':False, 'DC_KM_4':None, 'AC_KM_4':None,
                            'load_8':False, 'DC_KM_8':None, 'AC_KM_8':None,
                            'load_16':False, 'DC_KM_16':None, 'AC_KM_16':None,
                            'load_32':False, 'DC_KM_32':'/mnt/yifan/DC_KM1.pkl', 'AC_KM_32':'/mnt/yifan/AC_KM1.pkl',
                            'load_64':False, 'DC_KM_64':'/mnt/yifan/DC_KM1.pkl', 'AC_KM_64':'/mnt/yifan/AC_KM1.pkl',
                            'load_128':False, 'DC_KM_128':'/mnt/yifan/DC_KM1.pkl', 'AC_KM_128':'/mnt/yifan/AC_KM1.pkl',
                            'load_256':False, 'DC_KM_256':'/mnt/yifan/DC_KM1.pkl', 'AC_KM_256':'/mnt/yifan/AC_KM1.pkl',
                            'load_512':False, 'DC_KM_512':'/mnt/yifan/DC_KM1.pkl', 'AC_KM_512':'/mnt/yifan/AC_KM1.pkl',
                            'load_1024':False, 'DC_KM_1024':'/mnt/yifan/DC_KM1.pkl', 'AC_KM_1024':'/mnt/yifan/AC_KM1.pkl'}}

    print('--------------------start---------------------')
    print(par_old1)
    with open('dataset.json', 'r') as json_file:
        DataSet = json.load(json_file)
    Y = Load_from_Folder(folder=DataSet['CLIC_train_1024x1024_more'], color='RGB', ct=-1)+\
        Load_from_Folder(folder=DataSet['Holopix50k_trainR_1024x1024'], color='RGB', ct=2000)
    Y = np.array(Y).astype('float32')
    Y = downsample(Y, 4)
    print(Y.shape)
    gic = GIC(par_old1)
    gic.fit(Y)
    return gic

gic = run1()


from color_space import *
x = cv2.imread('lena512color.tiff')
x = BGR2RGB(x)
x = x.reshape(1, 512, 512, -1).astype('float32')
Y = downsample(x, 2)
stream, dc, save = gic.encode(Y)
iY = gic.decode(stream, dc)
print('MSE=%5.3f, PSNR=%3.5f'%(MSE(Y, iY), PSNR(Y, iY)))
print('--------------------end---------------------\n\n\n')
with open('dataset.json', 'r') as json_file:
    DataSet = json.load(json_file)
Y = Load_from_Folder(folder='/mnt/yifan/data/CLIC/test256/', color='RGB', ct=-1)
Y = np.array(Y).astype('float32')
stream, dc, save = gic.encode(Y)
iY = gic.decode(stream, dc)
print('MSE=%5.3f, PSNR=%3.5f'%(MSE(Y, iY), PSNR(Y, iY)))
print('--------------------end---------------------\n\n\n')