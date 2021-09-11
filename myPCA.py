# 2021.01.27
# @yifan & zhanxuan
# PCA transformation 
#
# 2D PCA modified from https://blog.csdn.net/w450468524/article/details/54895477
#
import numpy as np
import sklearn
from sklearn.decomposition import PCA

class myPCA():
    def __init__(self, n_components=-1, is2D=False, H=None, W=None):
        self.is2D         = is2D
        if is2D == False:
            self.n_components = n_components
            self.Kernels      = []
            self.PCA          = None
            self.Energy_ratio = []
            self.Energy       = []
            self.__version__  = sklearn.__version__ 
        else:     
            self.H            = H
            self.W            = W
            self.K1           = []
            self.K2           = []
    
    def PCA_2D_fit(self, X):
        # input: sample: (X, Y, Z)
        #                 X is the index of different block
        #                 Y and Z is the height and width of block
        #        row_top and col_top are the height and width of block
        # output:
        #        X: the horizontal kernel
        #        Z: the vertical kernel
        # Notes:
        #        forward transform: C = Z.T*A*X
        #        inverse transform: A = Z*C*X.T
        S = X.shape
        X = X.reshape(-1, self.W, self.H)
        mean = np.zeros((self.W, self.H))
        for i in range(X.shape[0]):
            mean = mean + X[i]
        mean /= float(X.shape[0])
        cov_row = np.zeros((self.H, self.H))
        for i in range(X.shape[0]):
            diff = X[i] - mean
            cov_row = cov_row + np.dot(diff.T, diff)
        cov_row /= float(X.shape[0])
        row_eval, row_evec = np.linalg.eig(cov_row)
        sorted_index = np.argsort(row_eval)
        self.K1 = np.array(row_evec[:,sorted_index[:-self.H-1 : -1]]) 
        cov_col = np.zeros((self.W, self.W))
        for i in range(X.shape[0]):
            diff = X[i] - mean
            cov_col += np.dot(diff,diff.T)
        cov_col /= float(X.shape[0])
        col_eval, col_evec = np.linalg.eig(cov_col)
        sorted_index = np.argsort(col_eval)
        self.K2 = np.array(col_evec[:,sorted_index[:-self.W-1 : -1]])
        
    def PCA_2D_transform(self, X, inv=False):
        res = []
        S = X.shape
        X = X.reshape(-1, self.W, self.H)
        for i in range(X.shape[0]):
            if inv == False:
                res.append(np.dot(np.transpose(self.K2), np.dot(X[i], self.K1)))
            else:
                res.append(np.dot(self.K2, np.dot(X[i], np.transpose(self.K1))))
        return np.array(res).reshape(S)
    
    def PCA_sklearn(self, X):
        self.PCA          = PCA(  n_components=self.n_components  )
        self.PCA.fit(X)
        self.Kernels      = self.PCA.components_
        self.Energy_ratio = self.PCA.explained_variance_ratio_
        self.Energy       = self.PCA.explained_variance_
        
    def fit(self, X, whichPCA='sklearn'):
        if self.is2D == True:
            self.PCA_2D_fit(X)
        else:
            X = X.reshape(  -1, X.shape[-1]  )
            if self.n_components == -1:
                self.n_components = X.shape[-1]
            if whichPCA == 'sklearn':
                self.PCA_sklearn(  X  )
            else:
                assert (False), "whichPCA only support 'sklearn' when is2D==False!"
        return self
            
    def transform(self, X):
        if self.is2D == False:
            S = (list)(X.shape)
            S[-1] = -1
            X = X.reshape(  -1, X.shape[-1]  )
            tX = self.PCA.transform(X)
            return tX.reshape(S)
        else:
            return self.PCA_2D_transform(X, inv=False)

    def ML_inverse_transform(self, Xraw, X):
        if self.is2D == False:
            if K is not None:
                return np.dot(  X, K  ), K
            llsr = LLSR(onehot=False)
            llsr.fit(X.reshape(-1, X.shape[-1]), Xraw.reshape(-1, X.shape[-1]))
            S = X.shape
            X = llsr.predict_proba(X.reshape(-1, X.shape[-1])).reshape(S)
            return X, llsr.weight
        else:
            res1 = []
            S = Xraw.shape
            Xraw = Xraw.reshape(-1, self.W, self.H)
            X = X.reshape(-1, self.W, self.H)
            for i in range(Xraw.shape[0]):
                res1.append( np.dot(Xraw[i], self.K1))
            res1 = np.array(res1)
            tmpX, tmpY = [], []
            for i in range(Xraw.shape[0]):
                tmpX.append(np.transpose(X[i]))
                tmpY.append(np.transpose(res1[i]))
            tmpX = np.array(tmpX).reshape(-1, self.W)
            tmpY = np.array(tmpY).reshape(-1, self.W)
            weight1, _, _, _ = np.linalg.lstsq(tmpX, tmpY, rcond=None)
            tmpX_p = np.matmul(tmpX, weight1).reshape(-1, self.H, self.W)
            tmpX = []
            for i in range(Xraw.shape[0]):
                tmpX.append(np.transpose(tmpX_p[i]))
            tmpX = np.array(tmpX).reshape(-1, self.H)
            weight2, _, _, _ = np.linalg.lstsq(tmpX, Xraw.reshape(-1, self.H), rcond=None)
            tmpX_p = np.matmul(tmpX, weight2)
            return tmpX_p.reshape(S)

    def inverse_transform(self, X):
        if self.is2D == False:
            S = (list)(X.shape)
            S[-1] = -1
            X = X.reshape(  -1, X.shape[-1]  )
            tX = self.PCA.inverse_transform(X)
            tX = tX.reshape(S)
            return tX
        else:
            return self.PCA_2D_transform(X, inv=True)