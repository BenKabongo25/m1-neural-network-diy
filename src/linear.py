# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Linear module

import numpy as np
import module
from enums import Initialization


class Linear(module.Module):

    def __init__(self, 
                input: int, 
                output: int, 
                bias: bool=True,
                initialization: int=Initialization.LECUN):
        '''
        :param input: size of each input sample
        :param output: size of each output sample
        :param bias : If set to False, the layer will not learn an additive bias. Default: True
        :param initialization : initialization mode
        '''
        super().__init__()
        self.input = input
        self.output = output
        self.bias = bias
        self._parameters = dict()
        self._gradient = dict()
        self._init_parameters(initialization)
        self.zero_grad()


    def _init_parameters(self, initialization: int):
        input, output = self.input, self.output
        shape = (input, output)

        self._parameters['b'] = np.zeros(output)

        if initialization == Initialization.ONE:
            self._parameters['W'] = np.ones(shape)
        elif initialization == Initialization.RANDOM:
            self._parameters['W'] = np.random.random(shape)
        elif initialization == Initialization.UNIFORM:
            self._parameters['W'] = np.random.uniform(-np.sqrt(1/input), np.sqrt(1/input), shape)
        elif initialization == Initialization.XAVIER:
            std = np.sqrt(2/(input + output))
            self._parameters['W'] = np.random.normal(0, std, shape)
        elif initialization == Initialization.LECUN:
            std = np.sqrt(1/input)
            self._parameters['W'] = np.random.normal(0, std, shape)
        else:
            self._parameters['W'] = np.zeros(shape)


    def set_parameters(self, parameters):
        assert parameters['W'].shape == self._parameters['W'].shape, ('The size of the parameters does not ' +
        'correspond to the expected size')
        assert parameters['b'].shape == self._parameters['b'].shape, ('The size of the parameters does not ' +
        'correspond to the expected size')
        self._parameters['W'] = parameters['W'].copy()
        self._parameters['b'] = parameters['b'].copy()

    
    def get_parameters(self):
        parameters = dict()
        parameters['W'] = self._parameters['W'].copy()
        parameters['b'] = self._parameters['b'].copy()
        return parameters


    def forward(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2, 'Data must be 2D matrices'
        assert X.shape[1] == self.input, ('The number of input data features must be equal ' +
        'to the number of module input features')
        self._save_data(X)
        if self.bias:
            return X @ self._parameters['W'] + self._parameters['b'] 
        return X @ self._parameters['W']


    def zero_grad(self):
        self._gradient['W'] = np.zeros_like(self._parameters['W'])
        self._gradient['b'] = np.zeros_like(self._parameters['b'])


    def update_parameters(self, gradient_step: float=1e-3):
        self._parameters['W'] -= gradient_step * self._gradient['W']
        self._parameters['b'] -= gradient_step * self._gradient['b']


    def _assert_backward(self, X: np.ndarray, delta: np.ndarray):
        assert len(X.shape) == 2, 'Data must be 2D matrices'
        assert len(delta.shape) == 2, 'Delta must be 2D matrices'
        assert X.shape[0] == delta.shape[0], 'The number of sample data must equal the number of delta values'
        assert X.shape[1] == self.input, ('The number of input data features must be equal ' +
        'to the number of module input features')
        assert delta.shape[1] == self.output, ('The number of expected delta values must be equal ' +
        'to the size of the module output')


    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        self._assert_backward(X, delta)
        self._gradient["W"] += X.T @ delta / len(X)
        self._gradient["b"] += delta.mean(axis=0)


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        self._assert_backward(X, delta)
        return delta @ self._parameters["W"].T
