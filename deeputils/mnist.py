from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def download_if_not_exist():
    mnist = input_data.read_data_sets("MNIST_data")
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, mnist.validation.images, mnist.validation.labels

def one_hot(labels_vec, nb_classes):
    return np.eye(nb_classes)[labels_vec]
    
def load_data():
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    # let's make sure we got the correct data
    assert x_train.shape == (55000, 28, 28, 1)

    y_train = one_hot(y_train, 10)
    assert y_train.shape == (55000, 10)

    x_test = x_test.reshape(-1, 28, 28, 1)
    assert x_test.shape == (10000, 28, 28, 1)

    y_test = one_hot(y_test, 10)
    assert y_test.shape == (10000, 10)

    x_val = x_val.reshape(-1, 28, 28, 1)
    assert x_val.shape == (5000, 28, 28, 1)

    y_val = one_hot(y_val, 10)
    assert y_val.shape == (5000, 10)

    return x_train, y_train, x_test, y_test, x_val, y_val
