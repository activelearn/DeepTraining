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
    pass

class Softmax(Activation):
    pass


class Loss(Activation):
    pass


class CrossEntropy(Loss):
    pass
