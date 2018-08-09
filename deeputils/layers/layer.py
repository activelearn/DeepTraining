from abc import ABCMeta, abstractmethod
import numpy as np
from layers.activation.activation import relu, sigmoid, softmax, leakyrelu, elu


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
        elif activation == "leakyrelu":
            self.activation = leakyrelu
        elif activation == "elu":
            self.activation = elu
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
    def __init__(self, nC_out, filter_size, input_shape, stride=1, activation, padding=0):
        assert len(filter_size) == 2
        assert len(input_shape) == 4

        if activation == "relu":
            self.activation = relu
        elif activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "leakyrelu":
            self.activation = leakyrelu
        elif activation == "elu":
            self.activation = elu
        else:
            raise NotImplementedError()

        self.pad = padding
        self.input_shape = input_shape
        self.stride = stride
        self.filter_size = filter_size
        self.nC = nC_out

        self.W = np.random.randn(
            filter_size[0], filter_size[1], input_shape[3], nC_out) * 0.01
        self.b = np.zeros(1, 1, 1, nC_out)

    def forward(self, X):
        assert(X.shape == self.input_shape)

        ns_vertical = (X.shape[1] - self.filter_size[0] + self.pad) / self.stride
        ns_horizontal = (X.shape[2] - self.filter_size[1] + self.pad) / self.stride

        out = np.zeros((X.shape[0], ns_vertical, ns_horizontal, self.nC))
        for i in range(ns_vertical):
            for j in range(ns_horizontal):
                start_v = i*self.stride
                start_h = j*self.stride
                #we'll vectorize it using numpy broadcasting rule
                out[:, i, j, :] = np.sum(X[:, start_v:start_v+self.filter_size[0], start_h:start_h+self.filter_size[1],:,np.newaxis]*self.W + self.b, axis=(1,2,3))

        return out, self.activation.forward(out)

    def _activation_backward(self, Z, backward_err):
        return backward_err * self.activation.backward(Z)

    def backward(self, X, Z, backward_err):
        dz = self._activation_backward(Z, backward_err)

    def update(self, grad, lr):
        (dw, db) = grad
        self.W -= lr * dW
        self.b -= lr * db


class MaxPooling2D(Layer):
    pass


class Dropout(Layer):
    pass
