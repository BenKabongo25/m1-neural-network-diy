# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Convolution 2D

import enums
import module
import numpy as np


class Conv2D(module.Module):

    def __init__(self, 
                k_width: int,
                k_height: int,
                chan_in: int, 
                chan_out: int, 
                stride_w: int=1,
                stride_h: int=1,
                bias: bool=True,
                initialization: int=enums.Initialization.LECUN):
        super().__init__()
        self.k_width = k_width
        self.k_height = k_height
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.bias = bias
        self._parameters = dict()
        self._gradient = dict()
        self._init_parameters(initialization)
        self.zero_grad()


    def _init_parameters(self, initialization: int):
        self._parameters['b'] = np.zeros(self.chan_out)

        def init():
            shape = (self.k_width, self.k_height, self.chan_in)
            if initialization == enums.Initialization.ONE:
                return np.ones(shape)
            if initialization == enums.Initialization.RANDOM:
                return np.random.random(shape)
            if initialization == enums.Initialization.UNIFORM:
                return np.random.uniform(-np.sqrt(1/input), np.sqrt(1/input), shape)
            if initialization == enums.Initialization.XAVIER:
                std = np.sqrt(2/(self.k_size + self.chan_in))
                return np.random.normal(0, std, shape)
            if initialization == enums.Initialization.LECUN:
                std = np.sqrt(1/self.k_size)
                return np.random.normal(0, std, shape)
            return np.zeros(shape)

        self._parameters['W'] = np.zeros((self.chan_out, self.k_size, self.chan_in))
        for i in range(self.chan_out):
            self._parameters['W'][i] = 1e-1 * init()
        

    def zero_grad(self):
        self._gradient['W'] = np.zeros_like(self._parameters['W'])
        self._gradient['b'] = np.zeros_like(self._parameters['b'])


    def set_parameters(self, parameters):
        assert parameters['W'].shape == self._parameters['W'].shape, ('The size of the parameters does not ' +
        'correspond to the expected size')
        assert parameters['b'].shape == self._parameters['b'].shape, ('The size of the parameters does not '
        'correspond to the expected size')
        self._parameters['W'] = parameters['W'].copy()
        self._parameters['b'] = parameters['b'].copy()


    def get_parameters(self):
        parameters = dict()
        parameters['W'] = self._parameters['W'].copy()
        parameters['b'] = self._parameters['b'].copy()
        return parameters


    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 4, 'Input data must be 4 dimensional'
        batch_size, width, height, chan_in = X.shape
        assert chan_in == self.chan_in, 'Input data channel count does not match expected channel count'

        idx = np.arange(out_length) * self.stride 
        X_col = (X_[:, 
            idx[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :], 
            :].reshape((batch_size, out_length, -1)))

        weight_col = self._parameters['W'].reshape(self.chan_out, -1)
        out = X_col @ weight_col.T
        if self.bias: out += self._parameters['b'].T
        out = out.reshape(batch_size, out_length, self.chan_out)

        self._X = X
        self._length_padded = length_padded
        self._padding_value = padding
        self._out_length = out_length
        self._X_col = X_col

        return out


    def _assert_backward(self, X: np.ndarray, delta: np.ndarray):
        assert X.ndim == 3, 'Input data must be 3 dimensional'
        assert X.shape[2] == self.chan_in, 'Input data channel count does not match expected channel count'
        assert delta.ndim == 3, 'Delta must be 3 dimensional'
        assert delta.shape[0] == X.shape[0], 'Batch size of delta and input must match'
        assert delta.shape[1] == self._out_length, 'Output length of data and output size of input must match'
        assert delta.shape[2] == self.chan_out, 'Output channel count of delta does not match expected channel count'


    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):    
        self._assert_backward(X, delta)
        delta_col = delta.transpose(2, 0, 1).reshape(self.chan_out, -1)
        self._gradient['W'] += ((delta_col @ self._X_col.reshape(-1, self.k_size * self.chan_in))
            .reshape(self._parameters['W'].shape)) / len(X)
        if self.bias: self._gradient['b'] += delta_col.mean(axis=1)


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        self._assert_backward(X, delta)
        batch_size = delta.shape[0]
        weight_col = self._parameters['W'].reshape(self.chan_out, -1)
        dX_col = (delta @ weight_col).reshape(batch_size, self._out_length, self.k_size, self.chan_in)
        dX = np.zeros((batch_size, self._length_padded, self.chan_in))
        idx = self.stride * np.arange(self._out_length)[:, None] + np.arange(self.k_size)
        dX[:, idx, :] += dX_col
        return dX[:, self._padding_value:-self._padding_value, :] if self.padding else dX
