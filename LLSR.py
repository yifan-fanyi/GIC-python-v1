# 2020.01.28
# @yifan
#
import numpy as np

class LLSR():
    def __init__(self, onehot=True, normalize=False):
        self.onehot = onehot
        self.normalize = normalize
        self.weight = []

    def fit(self, X, Y):
        if self.onehot == True:
            Y = np.eye(len(np.unique(Y)))[Y.reshape(-1)]
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        self.weight, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return self

    def predict(self, X):
        pred = self.predict_proba(X)
        return np.argmax(pred, axis=1)

    def predict_proba(self, X):
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        pred = np.matmul(X, self.weight)
        if self.normalize == True:
            pred = (pred - np.min(pred, axis=1, keepdims=True))/ np.sum((pred - np.min(pred, axis=1, keepdims=True) + 1e-15), axis=1, keepdims=True)
        return pred

    def score(self, X, Y):
        pred = self.predict(X)
        return accuracy_score(Y, pred)