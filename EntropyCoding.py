import numpy as np
import copy 
from Huffman import Huffman
from BinaryTree import BinaryTree

class EntropyCoding():
    def __init__(self, par):
        self.par = par
        self.bpp = 0.
        self.EC_list = {}

    def idx_select(self, label, idx=None):
        if idx is None:
            return label
        tmp = []
        r = label.shape[1] // idx.shape[1]
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                for k in range(label.shape[2]):
                    if idx[i, j//r, k//r] == 1:
                        tmp.append(label[i,j,k, 0])
        return np.array(tmp).astype('int32')

    def inverse_idx_select(self, tmp, idx, S):
        label = np.zeros(S)
        r = label.shape[1] // idx.shape[1]
        ct = 0
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                for k in range(label.shape[2]):
                    if idx[i, j//r, k//r] == 1:
                        label[i,j,k,0] = tmp[ct]
                        ct += 1
        return label

    def fit(self, save):
        label_list, idx_list = save['label'], save['idx']
        for i in range(1, self.par['n_hop']+1):
            label, idx = label_list['hop'+str(i)], idx_list['hop'+str(i)]
            tmp_list = [Huffman().fit(label[0])]
            for j in range(1, len(self.par['win']['hop'+str(i)])):
                tmp_list.append(Huffman().fit(self.idx_select(label[j], idx[j-1])))
            self.EC_list['hop'+str(i)] = tmp_list
        return self

    def refit(self, save, par):
        label_list, idx_list = save['label'], save['idx']
        for i in range(1, self.par['n_hop']+1):
            label, idx = label_list['hop'+str(i)], idx_list['hop'+str(i)]
            if par['hop'+str(i)][0][0] == True:
                self.EC_list['hop'+str(i)][0] = Huffman().fit(label[0])
            for j in range(1, len(self.par['win']['hop'+str(i)])):
                if par['hop'+str(i)][j][0] == True:
                    self.EC_list['hop'+str(i)][j] = Huffman().fit(self.idx_select(label[j], idx[j-1]))
        return self

    def encode(self, save, S=1024):
        self.bpp = 0.
        stream = {'idx':{}, 'label':{}}
        label_list, idx_list = save['label'], save['idx']
        n_img = label_list['hop1'][0].shape[0]
        for i in range(1, self.par['n_hop']+1):
            label, idx = label_list['hop'+str(i)], idx_list['hop'+str(i)]
            st = BinaryTree.saver(idx)
            stream['idx']['hop'+str(i)] = st
            print('Hop-%d'%(i))
            print('   save tree:%1.8f'%(len(st)/(n_img*S**2)))
            self.bpp += len(st)/(n_img*S**2)
            tmp_st = [self.EC_list['hop'+str(i)][0].encode(label[0])]
            print('  Level-0 Huffman %1.8f bpp'%((len(tmp_st[-1])) / (n_img*S**2)))
            self.bpp += len(tmp_st[-1]) / (n_img*S**2)
            for j in range(1, len(self.par['win']['hop'+str(i)])):
                st = self.EC_list['hop'+str(i)][j].encode(self.idx_select(label[j], idx[j-1]))
                tmp_st.append(st)
                print('  Level-%d Huffman %1.8f bpp'%(j, len(st)/(n_img*S**2)))
                self.bpp += len(st)/(n_img*S**2)
            stream['label']['hop'+str(i)] = tmp_st
        print('Bit rate %1.6f'%self.bpp)
        return stream

    def decode(self, stream):
        save = {'label':{}, 'idx':{}}
        for i in range(1, self.par['n_hop']+1):
            if len(self.par['shape']['hop'+str(i)]) > 1:
                save['idx']['hop'+str(i)] = BinaryTree.loader(stream['idx']['hop'+str(i)], self.par['shape']['hop'+str(i)][:-1])
            else:
                save['idx']['hop'+str(i)] = []
            tmp, _ = self.EC_list['hop'+str(i)][0].decode(stream['label']['hop'+str(i)][0])
            tmp = tmp.reshape(self.par['shape']['hop'+str(i)][0])
            label = [tmp]
            for j in range(1, len(self.par['win']['hop'+str(i)])):
                tmp, _ = self.EC_list['hop'+str(i)][j].decode(stream['label']['hop'+str(i)][j])
                tmp = self.inverse_idx_select(tmp, save['idx']['hop'+str(i)][j-1], self.par['shape']['hop'+str(i)][j])
                label.append(tmp)
            save['label']['hop'+str(i)] = label
        return save