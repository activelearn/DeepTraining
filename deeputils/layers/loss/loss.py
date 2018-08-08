from abc import ABCMeta, abstractmethod
import numpy as np


class Loss(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, X, Y):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, X, Y):
        raise NotImplementedError()


class CrossEntropy(Loss):
    def _softmax(self, X):
        expvx = np.exp(X - np.max(X, axis=1)[..., np.newaxis])
        return expvx/np.sum(expvx, axis=1, keepdims=True)

    def forward(self, X, Y):
        sf = self._softmax(X)
        return -np.log(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)]) / X.shape[0]

    def backward(self, X, Y):
        err = self._softmax(X)
        return (err - Y) / X.shape[0]


class MSE(Loss):
    pass


cross_entropy = CrossEntropy()
mse = MSE()
