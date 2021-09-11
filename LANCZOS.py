import numpy as np
from PIL import Image

class LANCZOS:
    @staticmethod
    def split(X, ratio=2):
        DC_L = LANCZOS.resample(X, ratio)
        X_inv_L = LANCZOS.resample(DC_L, 1/ratio)
        X_inv_L = np.round(X_inv_L)
        AC = X - X_inv_L
        return DC_L, AC
    @staticmethod
    def inv_split(DC, AC, ratio=2):
        DC = LANCZOS.resample(DC, 1/ratio)
        return DC+AC
    def inv_split_moreRef(DC, AC, k, i, j, win, ratio=2):
        DC = LANCZOS.resample(DC, 1/ratio)
        DC = Shrink(DC, win=win)
        return DC[k, i, j].reshape(-1)+AC.reshape(-1)
    @staticmethod
    def resample(X, ratio=2):
        image_list = []
        for i in range(X.shape[0]):
            tmp = []
            for k in range(X.shape[-1]):
                size1 = int(X.shape[1]/ratio)
                size2 = int(X.shape[2]/ratio)
                image_tmp = Image.fromarray(X[i,:,:,k]).resize(size=(size1, size2), resample=Image.LANCZOS)
                tmp.append(np.array(image_tmp).reshape(1, size1, size2, 1))
            image_list.append(np.concatenate(tmp, axis=-1))
        output = np.concatenate(image_list, axis=0)
        return output