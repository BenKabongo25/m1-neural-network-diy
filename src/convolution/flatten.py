# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Flatten

import module
import numpy as np


class Flatten(module.Module):

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return X.reshape(X.shape[0], -1)


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return delta.reshape(X.shape) * X


