from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, dZ):
        raise NotImplementedError()

    @abstractmethod
    def get_grad_backward(self, dZ, A):
        raise NotImplementedError()

    @abstractmethod
    def update(self, grad, lr):
        raise NotImplementedError()


class Dense(Layer):
    pass

class Conv2D(Layer):
    pass
    

class MaxPooling2D(Layer):
    pass


class Dropout(Layer):
    pass


class Flatten(Layer):
    pass
