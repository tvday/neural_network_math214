"""
Code taken and modified from work by Micheal Nielsen
https://github.com/mnielsen/neural-networks-and-deep-learning
http://neuralnetworksanddeeplearning.com/chap1.html
"""

import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNet(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        rng = np.random.default_rng()
        self.biases = [rng.normal(size=(n, 1)) for n in sizes[1:]]
        self.weights = [rng.normal(size=(n, m)) for n, m in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
            print("Start (random weights and biases): %d / %d" % (self.evaluate(test_data), n_test))
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            for j in range(0, n, mini_batch_size):
                mini_batch = training_data[j:j + mini_batch_size]
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch %d: %d / %d" % (i, self.evaluate(test_data), n_test))
            else:
                print("Epoch %d complete" % i)

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)
                       ]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # z = np.dot(w, activation) + b
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y
