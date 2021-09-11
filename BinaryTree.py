import copy
import numpy as np

class BinaryTree:
    @staticmethod
    def saver(idx):
        def idxConvert(idx):
            for i in range(1, len(idx)):
                r = idx[i].shape[1] // idx[i-1].shape[1]
                for k in range(idx[i].shape[0]):
                    for ii in range(idx[i].shape[1]):
                        for jj in range(idx[i].shape[2]):
                            if idx[i-1][k,ii//r,jj//r,0] < 0.5:
                                idx[i][k, ii, jj, 0] = -2 
            return idx
        stream = []
        idx = idxConvert(idx)
        for i in range(0, len(idx)):
            for k in range(idx[i].shape[0]):
                for ii in range(idx[i].shape[1]):
                    for jj in range(idx[i].shape[2]):
                        if idx[i][k,ii,jj,0] > -1:
                            stream.append(idx[i][k,ii,jj,0])
        return stream
    @staticmethod
    def loader(stream, S):
        idx, ct = [], 0
        for i in range(len(S)):
            idx.append(np.zeros(S[i]).astype('int16'))
        for k in range(idx[0].shape[0]):
            for ii in range(idx[0].shape[1]):
                for jj in range(idx[0].shape[2]):
                    idx[0][k,ii,jj,0] = stream[ct]
                    ct += 1
        for i in range(1, len(idx)):
            r = idx[i].shape[1] // idx[i-1].shape[1]
            for k in range(idx[i].shape[0]):
                for ii in range(idx[i].shape[1]):
                    for jj in range(idx[i].shape[2]):
                        if idx[i-1][k,ii//r,jj//r,0] == 0:
                            idx[i][k,ii,jj,0] = 0
                        else:
                            idx[i][k,ii,jj,0] = stream[ct]
                            ct += 1
        return idx
    @staticmethod
    def checker(idx, ref_idx):
        for i in range(len(idx)):
            s = np.sum(np.abs(idx[i]-ref_idx[i]))
            assert(s == 0), "Error!"
        
    @staticmethod        
    def UnitTest():
        def gen_data(S):
            idx = [np.random.randint(2, size=S[0])]
            for i in range(1,len(S)):
                r = S[i][1] // S[i-1][1]
                idx.append(np.random.randint(2, size=S[i]))
                for k in range(idx[i].shape[0]):
                    for ii in range(idx[i].shape[1]):
                        for jj in range(idx[i].shape[2]):
                            if idx[i-1][k,ii//r,jj//r, 0] == 0:
                                idx[i][k,ii,jj,0] = 0
            return idx
        S = [(5,4,4,1), (5,8,8,1),(5,16,16,1),(5,64,64,1)]
        idx = gen_data(S)
        stream = BinaryTree.saver(copy.deepcopy(idx))
        idxd = BinaryTree.loader(stream, S)
        BinaryTree.checker(idxd, idx)
        print("UnitTest Pass")