import numpy as np
from collections import deque
from layers.loss.loss import mse, cross_entropy

class Network(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, lr):
        self._lr = lr
        if loss == "cross_entropy":
            self._loss = cross_entropy
        elif loss == "mse":
            self._loss = mse
        else:
            raise NotImplementedError()

    def _compute(self, batch_x):
        out = batch_x
        for layer in self.layers:
            _, out = layer.forward(out)
        return out

    def _train(self, batch_x, batch_y):
        layer_inputs = deque([batch_x])
        curr_activation = batch_x

        for layer in self.layers:
            z, curr_activation = layer.forward(curr_activation)
            layer_inputs.appendleft(z)

        activation_backward = self._loss.backward(curr_activation, batch_y)
        grads = deque()

        for layer in reversed(self.layers):
            Z = layer_inputs.popleft()
            dX, dW, db = layer.backward(X=layer_inputs[0], Z=Z, backward_err=activation_backward)
            grads.appendleft((dW, db))
            activation_backward = dX

        for layer in self.layers:
            layer.update(grads.popleft(), self._lr)

        assert len(grads) == 0
        

    def fit(self, x_train, y_train, batch_size, epochs, validation_data):
        for epoch in range(epochs):
            shuffled = np.random.permutation(x_train.shape[0])
            batch_x_train = x_train[shuffled[:batch_size]]
            batch_y_train = y_train[shuffled[:batch_size]]
            self._train(batch_x_train, batch_y_train)

            loss = np.sum(cross_entropy.forward(self._compute(batch_x_train), batch_y_train))
            print("Epoch: %d, loss: %f" % (epoch, loss))

    def predict(self, X):
        pass

    def evaluate(self, x_test, y_test):
        predicted = self._compute(x_test)
        predicted = np.argmax(predicted, axis=1)
        labels = np.argmax(y_test, axis=1)
        accuracy = np.sum(predicted == labels) / x_test.shape[0] * 100
        print('Accuracy : %f' %accuracy)
