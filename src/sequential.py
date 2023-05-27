# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Sequential

import module as m
import linear
import numpy as np


class Sequential(m.Module):
    def __init__(self, *modules):
        self._modules = []
        self._last_linear_module_output = None
        for module in modules:
            self.add_module(module)


    def add_module(self, module: m.Module):
        assert isinstance(module, m.Module), 'The module object must be an instance of the Module class'
        
        if self._last_linear_module_output is not None:
            if isinstance(module, linear.Linear):
                assert self._last_linear_module_output == module.input, ('The output size of the last linear ' + 
                'module must be equal to the input size of the new module')
        
        if isinstance(module, linear.Linear):
            self._last_linear_module_output = module.output

        elif isinstance(module, Sequential):
            self._last_linear_module_output = module._last_linear_module_output

        self._modules.append(module)


    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        for module in self._modules:
            X = module.forward(X)
        return X


    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()


    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        for module in reversed(self._modules):
            module.backward_update_gradient(module._X, delta)
            delta = module.backward_delta(module._X, delta)


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        for module in reversed(self._modules):
            delta = module.backward_delta(module._X, delta)
        return delta


    def update_parameters(self, gradient_step: float = 0.001):
        for module in reversed(self._modules):
            module.update_parameters(gradient_step)
    

    def set_parameters(self, parameters):
        assert len(parameters) == len(self._modules), 'The number of parameters and the number of modules must match'
        for i in range(len(parameters)):
            self._modules[i].set_parameters(parameters[i])


    def get_parameters(self):
        parameters = []
        for module in self._modules:
            parameters.append(module.get_parameters())
        return parameters
