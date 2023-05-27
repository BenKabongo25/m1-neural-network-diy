# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# AutoEncoder


import activation
import linear
import module as m
import numpy as np
import sequential


class AutoEncoder(sequential.Sequential):

    def __init__(self, encoder: m.Module, decoder: m.Module):
        super().__init__(encoder, decoder)
        

    def get_encoder(self) -> m.Module:
        return self._modules[0]


    def get_decoder(self) -> m.Module:
        return self._modules[1]


    def parameters_sharing(self):
        def _set_parameters(parameters, module):
            if isinstance(module, linear.Linear):
                parameters_ = dict()
                parameters_['W'] = parameters['W'].T
                parameters_['b'] = module._parameters['b']
                module.set_parameters(parameters_)

            elif isinstance(module, sequential.Sequential):
                cpt = 0
                for parameters_ in parameters:
                    for i in range(cpt, len(module._modules)):
                        cpt += 1
                        if not isinstance(module._modules[i], activation.Activation):
                            _set_parameters(parameters_, module._modules[i])
                            break

        encoder_parameters = self._modules[0].get_parameters()
        encoder_parameters = list(filter(None, encoder_parameters))
        _set_parameters(reversed(encoder_parameters), self._modules[1])