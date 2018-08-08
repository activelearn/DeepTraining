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

class Sigmoid(Activation):
    pass


class Softmax(Activation):
    pass


relu = Relu()
sigmoid = Sigmoid()
softmax = Softmax()
