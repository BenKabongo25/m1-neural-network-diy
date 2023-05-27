# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Convolution 1D

import enums
import module
import math
import numpy as np


class Conv1D(module.Module):

    def __init__(self, 
                k_size: int, 
                chan_in: int, 
                chan_out: int, 
                stride: int=1,
                bias: bool=True,
                initialization: int=enums.Initialization.LECUN):

        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self.bias = bias
        self._parameters = dict()
        self._gradient = dict()
        self._init_parameters(initialization)
        self.zero_grad()


    def _init_parameters(self, initialization: int):
        self._parameters['b'] = np.zeros(self.chan_out)

        def init():
            shape = (self.k_size, self.chan_in)
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
        assert X.ndim == 3, 'Input data must be 3 dimensional'
        batch_size, length, chan_in = X.shape
        assert chan_in == self.chan_in, 'Input data channel count does not match expected channel count'
        self._X = X
        self._out_length = (length - self.k_size) // self.stride + 1

        self._X_col = (X[:, 
            (self.stride * np.arange(self._out_length))[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :], 
            :].reshape((batch_size, self._out_length, -1)))

        Z = self._X_col @ self._parameters['W'].reshape(self.chan_out, -1).T
        if self.bias: Z += self._parameters['b'].T
        Z = Z.reshape(batch_size, self._out_length, self.chan_out)
        return Z


    def _assert_backward(self, X: np.ndarray, delta: np.ndarray):
        assert X.ndim == 3, 'Input data must be 3 dimensional'
        assert X.shape[2] == self.chan_in, 'Input data channel count does not match expected channel count'
        assert delta.ndim == 3, 'Delta must be 3 dimensional'
        assert delta.shape[0] == X.shape[0], 'Batch size of delta and input must match'
        assert delta.shape[1] == self._out_length, 'Output length of data and output size of input must match'
        assert delta.shape[2] == self.chan_out, 'Output channel count of delta does not match expected channel count'


    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):    
        self._assert_backward(X, delta)
        batch_size = delta.shape[0]
        if self.bias: self._gradient['b'] += delta.mean(axis=(0, 1))
        self._gradient['W'] += (
            delta.transpose(2, 0, 1).reshape(self.chan_out, -1) @ 
            self._X_col.reshape(-1, self.k_size * self.chan_in)
        ).reshape(self._parameters['W'].shape) / batch_size


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        self._assert_backward(X, delta)
        batch_size = delta.shape[0]
        tmp = ( (delta @ self._parameters['W'].reshape(self.chan_out, -1)) / batch_size
            ).reshape(batch_size, self._out_length, self.k_size, self.chan_in)
        dX = np.zeros_like(X, dtype=float)
        dX[:, self.stride * np.arange(self._out_length)[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :]] += tmp[:, np.arange(self._out_length)]
        return dX


class ConvTranspose1D(Conv1D):

    def __init__(self, 
                k_size: int, 
                chan_in: int, 
                chan_out: int, 
                stride: int=1,
                bias: bool=True,
                width: int=None,
                height: int=None,
                initialization: int=enums.Initialization.LECUN):

        super().__init__(k_size, chan_in, chan_out, stride, bias, initialization)
        self.width = width
        self.height = height if height is not None else width


    def _compute_dimensions(self, length):
        width, height = 0, 0
        if self.width is not None:
            assert length == self.width * self.height, 'The length of the image does not match the expected length'
            width = self.width
            height = self.height
        else:
            d = int(math.sqrt(length))
            assert d ** 2 == length, 'The width and height of the image are not known'
            width = d
            height = d
        return width, height


    def forward(self, X: np.ndarray) -> np.ndarray:
        batch_size, length, chan_in = X.shape
        width, height = self._compute_dimensions(length)
        X_transposed = ( X.reshape(batch_size, width, height, chan_in)
                .transpose((0, 2, 1, 3))
                .reshape(batch_size, width * height, chan_in))
        return super().forward(X_transposed)


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        batch_size, length, chan_in = X.shape
        width, height = self._compute_dimensions(length)
        X_transposed = ( X.reshape(batch_size, width, height, chan_in)
                .transpose((0, 2, 1, 3))
                .reshape(batch_size, width * height, chan_in))
        dX = super().backward_delta(X_transposed, delta)
        return (  dX.reshape(batch_size, height, width, chan_in)
                    .transpose((0, 2, 1, 3))
                    .reshape(batch_size, height * width, chan_in))


class DoubleConv1D(module.Module):

    def __init__(self, 
                k_size: int, 
                chan_in: int, 
                chan_out: int, 
                stride: int=1,
                bias: bool=True,
                width: int=None,
                height: int=None,
                initialization: int=enums.Initialization.LECUN):

        self._conv = Conv1D(k_size, chan_in, chan_out, stride, bias, initialization)
        self._conv_transpose = ConvTranspose1D(k_size, chan_in, chan_out, stride, bias, width, height, initialization)


    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        Z1 = self._conv(X)
        Z2 = self._conv_transpose(X)
        return np.concatenate((Z1, Z2), axis=0)

    
    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        self._conv.backward_delta(X, delta)
        self._conv_transpose.backward_delta(X, delta)


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        dX1 = self._conv.backward_delta(X, delta[:self._conv._out_length])
        dX2 = self._conv_transpose.backward_delta(X, delta[self._conv._out_length:])
        dX = (dX1 + dX2) / 2
        return dX


    def update_parameters(self, gradient_step: float = 0.001):
        self._conv.update_parameters(gradient_step)
        self._conv_transpose.update_parameters(gradient_step)


    def get_parameters(self):
        parameters = []
        parameters.append(self._conv.get_parameters())
        parameters.append(self._conv_transpose.get_parameters())
        return parameters


    def set_parameters(self, parameters):
        self._conv.set_parameters(parameters[0])
        self._conv_transpose.set_parameters(parameters[1])


