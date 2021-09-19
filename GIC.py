import numpy as np
import os
import pickle
import copy 
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
from load_img import Load_from_Folder, Load_Images
from evaluate import Time, MSE, PSNR
from MBMBVQ import MBMBVQ
from EntropyCoding import EntropyCoding

class GIC():
    def __init__(self, par):
        self.MBMBVQ = MBMBVQ(par)
        self.EC = EntropyCoding(par)
    def change_n_img(self, n_img):
        for i in range(1, self.EC.par['n_hop']+1):
            for j in range(len(self.EC.par['shape']['hop'+str(i)])):
                self.EC.par['shape']['hop'+str(i)][j][0] = n_img
    @Time
    def fit(self, Y):
        self.change_n_img(Y.shape[0])
        self.MBMBVQ.fit(copy.deepcopy(Y))
        save = self.MBMBVQ.encode(copy.deepcopy(Y))
        self.EC.fit(save)
        return self
    @Time
    def refit(self, Y, par):
        self.change_n_img(Y.shape[0])
        self.MBMBVQ.refit(copy.deepcopy(Y), par)
        save = self.MBMBVQ.encode(copy.deepcopy(Y))
        self.EC.refit(save, par)
        return self
    @Time
    def encode(self, Y):
        self.change_n_img(Y.shape[0])
        save = self.MBMBVQ.encode(Y)
        stream = self.EC.encode(save, S=Y.shape[1])
        return stream, save['DC'], save
    
    @Time
    def decode(self, stream, DC):
        save = self.EC.decode(stream)
        save['DC'] = DC
        iY = self.MBMBVQ.decode(save)
        return iY

    # return pickleable obj
    def save(self):
        for k in self.MBMBVQ.km.keys():
            km = self.MBMBVQ.km[k]
            for i in km.KM:
                i.KM.KM = None
                i.KM.saveObj=False
        return self
   

if __name__ == "__main__":
    with open('./test_data/test_par1.json', 'r') as f:
        par = json.load(f)
    gic = GIC_Y(par)

    Y_list = Load_from_Folder(folder='./test_data/', color='YUV', ct=-1)
    Y = np.array(Y_list)[:,:,:,:1]

    gic.fit(Y)

    stream, dc = gic.encode(Y)

    iY = gic.decode(stream, dc)
    print('MSE=%5.3f, PSNR=%3.5f'%(MSE(Y, iY), PSNR(Y, iY)))
    print('------------------')
    print(" * Ref result: "+'MSE=129.342, PSNR=27.01340')
 