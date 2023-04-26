# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Utils

import numpy as np


def onehot_encoding(y, n_classes):
    onehot = np.zeros((y.size, n_classes))
    onehot[np.arange(y.size), y] = 1
    return onehot


def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)
