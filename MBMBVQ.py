import numpy as np
import copy 
from LANCZOS import LANCZOS
from util import Shrink, invShrink
from evaluate import Time, MSE, PSNR
from myKMeans import myKMeans
from myPCA import myPCA
from Huffman import Huffman
from BinaryTree import BinaryTree

class PCAKM():
    def __init__(self, n_cluster, n_components=None, gpu=False):
        self.n_components = n_components
        self.KM = myKMeans(n_clusters=n_cluster, fast=True, gpu=gpu, n_threads=8, saveObj=True)
        
    
    def fit(self, res):
        if self.n_components is not None:
            self.pca = myPCA(self.n_components).fit(res)
            res = self.pca.transform(res)
        self.KM.fit(res)
        return self

    def predict(self, res):
        if self.n_components is not None:
            res = self.pca.transform(res)
        label = self.KM.predict(res)
        return label

    def inverse_predict(self, label):
        iX = self.KM.inverse_predict(label)
        if self.n_components is not None:
            iX = self.pca.inverse_transform(iX)
        return iX
            
class MBVQ():
    def __init__(self, n_cluster, win, MSE_TH, n_components):
        self.win = win
        self.n_cluster = n_cluster
        self.KM = []
        self.MSE_TH = MSE_TH
        self.n_components = n_components
        self.PCA = []
        self.channel = 1

    def fit(self, X, gpu=False):
        self.channel = X.shape[-1]
        print('MBVQ channel',self.channel)
        res = copy.deepcopy(X)
        res = Shrink(res, self.win[0]).reshape(-1, self.channel*self.win[0]**2)
        km = PCAKM(self.n_cluster[0], self.n_components[0], gpu=gpu)
        km.fit(res)
        self.KM.append(km)
        iX = km.inverse_predict(km.predict(res))
        res = res - iX
        res = res.reshape(res.shape[0], self.channel, self.win[0], self.win[0])  
        res = np.moveaxis(res, 1, -1)
        for i in range(1, len(self.win)):
            tmp = []
            for ii in range(res.shape[0]):
                if MSE(res[ii], np.zeros_like(res[ii])) > self.MSE_TH[i]:
                    tmp.append(res[ii])    
            res = Shrink(np.array(tmp), self.win[i]).reshape(-1, self.channel*self.win[i]**2)
            km = PCAKM(self.n_cluster[i], self.n_components[i])
            km.fit(res)
            self.KM.append(km)
            iX = km.inverse_predict(km.predict(res))
            res = res - iX
            res = res.reshape(res.shape[0], self.channel, self.win[i], self.win[i])
            res = np.moveaxis(res, 1, -1)
        return self
    
    def refit(self, X, par, gpu=False):
        self.channel = X.shape[-1]
        print('MBVQ channel',self.channel)
        res = copy.deepcopy(X)
        res = Shrink(res, self.win[0]).reshape(-1, self.channel*self.win[0]**2)
        if par[0][0] == True:
            print('   MBVQ refit', 0, par[0][1])
            km = PCAKM(par[0][1], self.n_components[0], gpu=gpu)
            km.fit(res)
            self.KM[0] = km
        iX = self.KM[0].inverse_predict(self.KM[0].predict(res))
        res = res - iX
        res = res.reshape(res.shape[0], self.channel, self.win[0], self.win[0])  
        res = np.moveaxis(res, 1, -1)
        for i in range(1, len(self.win)):
            tmp = []
            for ii in range(res.shape[0]):
                if MSE(res[ii], np.zeros_like(res[ii])) > self.MSE_TH[i]:
                    tmp.append(res[ii])    
            res = Shrink(np.array(tmp), self.win[i]).reshape(-1, self.channel*self.win[i]**2)
            if par[i][0] == True:
                print('   MBVQ refit', i, par[i][1])
                km = PCAKM(par[i][1], self.n_components[i])
                km.fit(res)
                self.KM[i] = km 
            iX = self.KM[i].inverse_predict(self.KM[i].predict(res))
            res = res - iX
            res = res.reshape(res.shape[0], self.channel, self.win[i], self.win[i])
            res = np.moveaxis(res, 1, -1)
        return self

    def select(self, X, k):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(-1, X.shape[-1])
        idx = []
        for i in range(X.shape[0]):
            if MSE(X[i], np.zeros_like(X[i])) > self.MSE_TH[k]:
                idx.append(1)
            else:
                idx.append(0)
        idx = np.array(idx).reshape(S)
        return idx
    
    def inv_select(self, X, idx, ratio):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    if idx[i,j//ratio, k//ratio, 0] < 0.5:
                        X[i,j,k] *= 0
        return X
    
    def idx_continuous(self, idx):
        for i in range(1, len(idx)):
            ratio = (int)(self.win[i-1] / self.win[i])
            for ii in range(idx[i].shape[0]):
                for jj in range(idx[i].shape[1]):
                    for kk in range(idx[i].shape[2]):
                        if idx[i-1][ii, jj//ratio, kk//ratio] == 0:
                            idx[i][ii,jj,kk] = 0
        return idx
    
    def predict(self, X):
        label, idx = [], []
        res = Shrink(copy.deepcopy(X), self.win[0])
        l = self.KM[0].predict(res)
        label.append(l)
        iX = self.KM[0].inverse_predict(l)
        res = invShrink(res - iX, self.win[0])
        for i in range(1, len(self.win)):
            idx_ = self.select(Shrink(copy.deepcopy(res), self.win[i-1]), i)
            idx.append(idx_)
            res = Shrink(res, self.win[i])
            l = self.KM[i].predict(res)
            label.append(l)
            iX = self.KM[i].inverse_predict(l)
            res = invShrink(res-iX, self.win[i])
        return label, self.idx_continuous(idx)
    
    def inverse_predict(self, label, idx):
        iX = self.KM[0].inverse_predict(label[0])
        iX = invShrink(iX, self.win[0])
        for i in range(1, len(self.win)):
            tmp = self.KM[i].inverse_predict(label[i])
            tmp = self.inv_select(tmp, idx[i-1], (int)(self.win[i-1]/self.win[i]))
            iX += invShrink(tmp, self.win[i])
        return iX
    
class MBMBVQ():
    def __init__(self, par):
        self.par = par
        self.km = {}
        self.save = {'label':{},
                     'idx':{}}

    @Time  
    def foreward(self, Y):
        DC, AC = {'hop0':Y}, {}
        DC_last = copy.deepcopy(Y)
        for i in range(1, self.par['n_hop']+1):
            tDC, tAC = LANCZOS.split(DC_last)
            DC['hop'+str(i)] = tDC
            AC['hop'+str(i)] = tAC
            DC_last = tDC
        return DC, AC
    
    @Time
    def backward(self, ACn, DCn, DCn_1, myhash, fit, gpu=False):
        print(myhash)
        if fit == True:
            self.km[myhash] = MBVQ(self.par['n_clusters'][myhash], 
                                          self.par['win'][myhash], 
                                          self.par['MSE_TH'][myhash],
                                          self.par['n_components'][myhash]).fit(ACn, gpu)
        labeln, idxn = self.km[myhash].predict(ACn)
        iACn = self.km[myhash].inverse_predict(labeln, idxn)
        iDCn_1 = LANCZOS.inv_split(DCn, iACn)
        err = DCn_1 - iDCn_1
        self.save['label'][myhash] = labeln
        self.save['idx'][myhash] = idxn
        return iDCn_1, err
        
    def fit(self, Y, gpu=False):
        DC, AC = self.foreward(Y)
        self.save = {'label':{},
                     'idx':{},
                     'DC':DC['hop'+str(self.par['n_hop'])]}
        newAC = AC['hop'+str(self.par['n_hop'])]
        iDC = DC['hop'+str(self.par['n_hop'])]
        for i in range(self.par['n_hop'], 1, -1):
            iDC, err = self.backward(newAC, 
                                     iDC,
                                     DC['hop'+str(i-1)],
                                     'hop'+str(i),
                                     fit=True,
                                     gpu=gpu)
            newAC = LANCZOS.inv_split(err, AC['hop'+str(i-1)])
        # Hop1
        print('hop1')
        self.km['hop1'] = MBVQ(self.par['n_clusters']['hop1'], 
                          self.par['win']['hop1'], 
                          self.par['MSE_TH']['hop1'],
                          self.par['n_components']['hop1']).fit(newAC, gpu=gpu)
        label1, idx1 = self.km['hop1'].predict(newAC)
        iAC1 = self.km['hop1'].inverse_predict(label1, idx1)
        iX = LANCZOS.inv_split(iDC, iAC1) 
        print('\nTraining MSE:',MSE(iX, Y))
        return self
    
    @Time
    def rebackward(self, ACn, DCn, DCn_1, myhash, fit, gpu=False):
        print(myhash)
        self.km[myhash] = self.km[myhash].refit(ACn, fit, gpu)
        labeln, idxn = self.km[myhash].predict(ACn)
        iACn = self.km[myhash].inverse_predict(labeln, idxn)
        iDCn_1 = LANCZOS.inv_split(DCn, iACn)
        err = DCn_1 - iDCn_1
        self.save['label'][myhash] = labeln
        self.save['idx'][myhash] = idxn
        return iDCn_1, err

    def refit(self, Y, par, gpu=False):
        DC, AC = self.foreward(Y)
        self.save = {'label':{},
                     'idx':{},
                     'DC':DC['hop'+str(self.par['n_hop'])]}
        newAC = AC['hop'+str(self.par['n_hop'])]
        iDC = DC['hop'+str(self.par['n_hop'])]
        for i in range(self.par['n_hop'], 1, -1):
            iDC, err = self.rebackward(newAC, 
                                     iDC,
                                     DC['hop'+str(i-1)],
                                     'hop'+str(i),
                                     fit=par['hop'+str(i)],
                                     gpu=gpu)
            newAC = LANCZOS.inv_split(err, AC['hop'+str(i-1)])
        # Hop1
        print('hop1')
        self.km['hop1'] = self.km['hop1'].refit(newAC, par['hop1'], gpu=gpu)
        label1, idx1 = self.km['hop1'].predict(newAC)
        iAC1 = self.km['hop1'].inverse_predict(label1, idx1)
        iX = LANCZOS.inv_split(iDC, iAC1) 
        print('\nTraining MSE:',MSE(iX, Y))
        return self

    def encode(self, Y):
        DC, AC = self.foreward(Y)
        self.save = {'label':{},
                     'idx':{},
                     'DC':DC['hop'+str(self.par['n_hop'])]}
        newAC = AC['hop'+str(self.par['n_hop'])]
        iDC = DC['hop'+str(self.par['n_hop'])]
        for i in range(self.par['n_hop'], 1, -1):
            iDC, err = self.backward(newAC, 
                                     iDC,
                                     DC['hop'+str(i-1)],
                                     'hop'+str(i),
                                     fit=False)
            newAC = LANCZOS.inv_split(err, AC['hop'+str(i-1)])
        # Hop1
        print('hop1')
        label1, idx1 = self.km['hop1'].predict(newAC)
        self.save['label']['hop1'] = label1
        self.save['idx']['hop1'] = idx1
        iAC1 = self.km['hop1'].inverse_predict(label1, idx1)
        iX = LANCZOS.inv_split(iDC, iAC1) 
        print('\nTesting MSE:',MSE(iX, Y))        
        return self.save

    def decode(self, save):
        iX = save['DC']
        label = save['label']
        idx = save['idx']
        for i in range(self.par['n_hop'], 0, -1):
            itX = self.km['hop'+str(i)].inverse_predict(label['hop'+str(i)], idx['hop'+str(i)])
            iX = LANCZOS.resample(iX, ratio=1/2) + itX
        return iX
    
    def print_shape(self, save):
        for k in save['label'].keys():
            print(k)
            for l in save['label'][k]:
                print(l.shape)