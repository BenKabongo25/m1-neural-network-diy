# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Average Pool 1D

import module
import numpy as np


class AvgPool1D(module.Module):

    def __init__(self, 
                k_size: int, 
                stride: int):
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    
    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 3, 'Input data must be 3 dimensional'
        batch_size, length, chan = X.shape
        self._X = X
        self._out_length = (length - self.k_size) // self.stride + 1

        X_col = (X[:, 
            (self.stride * np.arange(self._out_length))[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :], 
            :].reshape((batch_size, self._out_length, self.k_size, chan)))

        Z = X_col.mean(axis=2)
        return Z


    def _assert_backward(self, X: np.ndarray, delta: np.ndarray):
        assert X.ndim == 3, 'Input data must be 3 dimensional'
        assert delta.ndim == 3, 'Delta must be 3 dimensional'
        assert delta.shape[0] == X.shape[0], 'Batch size of delta and input must match'
        assert delta.shape[1] == self._out_length, 'Output length of data and output size of input must match'
        assert X.shape[2] == delta.shape[2], 'Output channel count of delta does not match output channel of input'


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        self._assert_backward(X, delta)
        batch_size, _, chan = X.shape
        tmp = (np.tile(delta, 2) / self.k_size).reshape(batch_size, self._out_length, self.k_size, chan) / batch_size
        dX = np.zeros_like(X, dtype=float)
        dX[:, self.stride * np.arange(self._out_length)[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :]] += tmp[:, np.arange(self._out_length)]
        return dX

