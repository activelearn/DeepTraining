import mnist
import numpy as np
from models.network import Network
from layers.layer import Dense

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, x_val, y_val = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    neuralnets = Network()
    neuralnets.add(Dense(shape=(x_train.shape[1], 200), activation="elu"))
    neuralnets.add(Dense(shape=(200, 10), activation="elu"))
    neuralnets.compile(loss="cross_entropy", lr=0.01)
    neuralnets.fit(x_train, y_train, 200, 20000, 0)

    neuralnets.evaluate(x_test, y_test)
