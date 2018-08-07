import numpy as np
from collections import deque


class Network(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, lr):
        self._loss = loss
        self._lr = lr

    def fit(self, x_train, y_train, batch_size, epochs, validation_data):
        pass

    def predict(self, X):
        pass

    def evaluate(self, x_test, y_test):
        pass
