import mnist    # from https://github.com/hsjeong5/MNIST-for-Numpy
import numpy as np
import matplotlib.pyplot as plt
from neuralnet.neuralnet import NeuralNet


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_images(images, n):
    """
    prints first n images
    :param images:
    :param n:
    :return:
    """
    for i in range(n):
        # prints the ith image in the training set
        img = images[i, :].reshape(28, 28)  # ith image in the training set.
        plt.imshow(img, cmap='gray')
        plt.show()  # Show the image


def prep_mnist_data():
    x_train, y_train, x_test, y_test = mnist.load()
    train_inputs = [np.reshape(x / 255, [784, 1]) for x in x_train]
    test_inputs = [np.reshape(x / 255, [784, 1]) for x in x_test]

    def vectorized(y):
        v = np.zeros((10, 1))
        v[y] = 1.0
        return v

    train_results = [vectorized(y) for y in y_train]

    training_data = [i for i in zip(train_inputs, train_results)]
    testing_data = [i for i in zip(test_inputs, y_test)]

    return training_data, testing_data


def train_mnist(training_data, sizes, epochs, mini_batch_size, learning_rate, testing_data=None):
    nn = NeuralNet(sizes)
    nn.sgd(training_data, epochs, mini_batch_size, learning_rate, testing_data)
    return nn


def example():
    """For the example in the slides."""

    a0 = np.array([0.25, 0.75, 0.50, 0.10, 0.85])
    w0 = np.array([[-5, 7.5, 0.2, 3, 4],
                   [4, -10, -0.5, -7, 0.6],
                   [6, 3, 0.1, 9, -4.4]])
    b0 = np.array([-15, 8, 5])

    a1 = sigmoid(w0 @ a0 + b0)
    w1 = np.array([[7, 8, 9],
                   [2, 0.7, 0.3],
                   [-2, 0.8, 11.2]])
    b1 = np.array([5, -1, 4])

    a2 = sigmoid(w1 @ a1 + b1)
    w2 = np.array([[-6, 11, 9.6]])
    b2 = np.array(15)

    a3 = sigmoid(w2 @ a2 + b2)

    result = a3


if __name__ == '__main__':
    training_data, testing_data = prep_mnist_data()
    mnist_nn = train_mnist(training_data, [784, 30, 15, 10], 30, 10, 2, testing_data)

