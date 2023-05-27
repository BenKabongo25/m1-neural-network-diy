# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Activations modules

import numpy as np
import module
import utils


class Activation(module.Module):
    def __init__(self):
        super().__init__()


    def _backward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return delta * self._backward(X)


class TanH(Activation):
    def __init__(self):
        super().__init__()


    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.tanh(X)


    def _backward(self, X: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(X) ** 2


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return 1 / (1 + np.exp(-X))

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.exp(-X) / ((1 + np.exp(-X)) ** 2)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.maximum(X, 0)

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, 0)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.maximum(X, self.alpha*X)

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, self.alpha)


class ELU(Activation):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.where(X > 0, X, self.alpha * (np.exp(X) - 1))

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, self.alpha * np.exp(X))


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return utils.softmax(X)

    def _backward(self, X: np.ndarray) -> np.ndarray:
        s = self.forward(X)
        return s * (1 - s)
        