from abc import ABCMeta, abstractmethod
import numpy as np
from layers.activation.activation import relu, sigmoid, softmax


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, X, Z, backward_err):
        raise NotImplementedError()

    @abstractmethod
    def update(self, grad, lr):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, shape, activation):
        assert len(shape) == 2
        self.shape = shape
        self.W = np.random.randn(shape[0], shape[1]) * 0.01
        self.b = np.zeros((1, shape[1]))
        if activation == "relu":
            self.activation = relu
        elif activation == "sigmoid":
            self.activation = sigmoid
        else:
            raise Exception("Activation function not recognized exception")

    def forward(self, X):
        z = X.dot(self.W) + self.b
        return z, self.activation.forward(z)

    def _activation_backward(self, Z, backward_err):
        return backward_err * self.activation.backward(Z)

    def backward(self, X, Z, backward_err):
        dZ = self._activation_backward(Z, backward_err)
        dX = dZ.dot(self.W.T)
        dW = X.T.dot(dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        return dX, dW, db

    def update(self, grad, lr):
        (dW, db) = grad
        self.W -= lr * dW
        self.b -= lr * db


class Flatten(Layer):
    pass


class Conv2D(Layer):
    def __init__(self, num_channels, kernel_size, activation, shape):
        assert len(kernel_size) == 2

        self.nC = num_channels
        self.kernel_size = kernel_size
        if activation == 'relu':
            self.activation = relu
        elif activation == 'softmax':
            self.activation = softmax
        elif activation == 'sigmoid':
            self.activation = sigmoid
        else:
            raise NotImplementedError()
        self.filters = np.random.randn(self.kernel_size) * 0.01
        self.shape = shape

    def forward(self, X):
        pass

    def backward(self, X, Z, backward_err):
        pass

    def update(self, grad, lr):
        pass
        
class MaxPooling2D(Layer):
    pass


class Dropout(Layer):
    pass
