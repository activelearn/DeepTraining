from abc import ABCMeta, abstractmethod
import numpy as np


class Activation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, inputs):
        raise NotImplementedError()


class Relu(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return 1. * (x > 0)


class LeakyRelu(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha*x, x)

    def backward(self, x):
        bw = 1. * (x > 0)
        bw[bw == 0] = self.alpha
        return bw


class ELU():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        out = x * (x >= 0)
        out[out == 0] = self.alpha * (np.exp(x[out == 0]) - 1)
        return out

    def backward(self, x):
        bw = 1.0 * (x >= 0)
        bw[bw == 0] = self.alpha * np.exp(x[bw == 0])
        return bw


class Sigmoid(Activation):
    pass


class Softmax(Activation):
    pass


relu = Relu()
sigmoid = Sigmoid()
softmax = Softmax()
leakyrelu = LeakyRelu()
elu = ELU()
