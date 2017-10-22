from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        exp_uk = np.exp(input)
        y_k = np.divide(exp_uk.T, np.sum(exp_uk, axis=1)).T
        return np.mean(np.sum(np.multiply(y_k, target), axis=1))

    def backward(self, input, target):
        #input ?
        return (input - target) / len(input)
