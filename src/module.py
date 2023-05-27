# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Module

import numpy as np


class Module(object):

    def __init__(self):
        self._parameters = None
        self._gradient   = None
        self._X          = None


    def zero_grad(self):
        '''
        set the gradient to 0
        '''
        pass


    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Compute the forward pass
        :param X: input data
            shape (m, d)
        '''
        raise NotImplementedError


    def _save_data(self, X: np.ndarray):
        '''
        Save forward data
        :param X: input data
            shape (m, d)
        '''
        self._X = X


    def __call__(self, X: np.ndarray):
        return self.forward(X)


    def update_parameters(self, gradient_step: float=1e-3):
        '''
        Calculation of the update of the parameters according to the calculated gradient 
        and the step of gradient_step
        :param gradient_step : learning rate
        '''
        pass


    def set_parameters(self, parameters):
        '''
        Set module parameters
        :param parameters
        '''
        pass

    
    def get_parameters(self):
        '''
        Get module parameters
        '''
        return None


    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        '''
        Update the gradient value
        :param X: input data
            shape (m, d)
        :param delta: current layer delta
            shape (m, d')
        '''
        pass


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        '''
        Calculate the derivative of the error
        :param X: input data
            shape (m, d)
        :param delta: current layer delta
            shape (m, d')
        :return previous layer delta
            shape (m, d)
        '''
        raise NotImplementedError
