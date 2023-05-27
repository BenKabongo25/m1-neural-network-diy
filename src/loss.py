# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Loss modules

import numpy as np
import utils


class Loss(object):
    def _assert_shape(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, 'The ground truth and prediction matrices must be of the same size'

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        return self.forward(y, yhat)

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        pass

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        pass


class MSELoss(Loss):
    '''
    MSE Loss
    '''

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Forward
        :param y: ground truth
            shape = (m,)
        :param yhat: prediction
            shape = (m, 1) or (m,)
        :return MSE
            shape = (m,)
        '''
        if y.ndim == 1: y = y.reshape((-1, 1))
        if yhat.ndim == 1: yhat = yhat.reshape((-1, 1))
        self._assert_shape(y, yhat)
        return (y - yhat)**2

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Backward
        :param y: ground truth
            shape = (m,)
        :param yhat: prediction
            shape = (m,)
        :return MSE grad
            shape = (m, 1)
        '''
        if y.ndim == 1: y = y.reshape((-1, 1))
        if yhat.ndim == 1: yhat = yhat.reshape((-1, 1))
        self._assert_shape(y, yhat)
        return (-2 * (y - yhat))
        

class BCELoss(Loss):
    '''
    Binary Cross Entropy Loss
    '''
    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Forward
        :param y: ground truth
            shape = (m,)
        :param yhat: prediction
            shape = (m,)
        :return BCE Loss
            shape = (m,)
        '''
        if y.ndim == 1: y = y.reshape(-1, 1)
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1-1e-12)
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Backward
        :param y: ground truth
            shape = (m,)
        :param yhat: prediction
            shape = (m,)
        :return BCE grad
            shape = (m, 1)
        '''
        if y.ndim == 1: y = y.reshape(-1, 1)
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1-1e-12)
        return (-y/yhat) + ((1 - y)/(1 - yhat))


class CELoss(Loss):
    '''
    Cross Entropy Loss With Softmax
    '''

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Forward
        :param y: ground truth
            shape = (m, k)
        :param yhat: prediction
            shape = (m, k)
        :return CE Loss
            shape = (m,)
        '''
        self._assert_shape(y, yhat)
        return -np.log(utils.softmax(yhat)[np.arange(len(y)), y.argmax(axis=1).reshape(-1)])

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Backward
        :param y: ground truth
            shape = (m, k)
        :param yhat: prediction
            shape = (m, k)
        :return CE grad
            shape = (m, k)
        '''
        return -y + utils.softmax(yhat)


class CCELoss(Loss):
    '''
    Categorical Cross Entropy Loss
    '''

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Forward
        :param y: ground truth
            shape = (m, k)
        :param yhat: prediction
            shape = (m, k)
        :return CCE Loss
            shape = (m,)
        '''
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1-1e-12)
        return -np.sum(y*np.log(yhat), axis=1)

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Backward
        :param y: ground truth
            shape = (m, k)
        :param yhat: prediction
            shape = (m, k)
        :return CCE grad
            shape = (m, k)
        '''
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1-1e-12)
        return -y/yhat


class HingeLoss(Loss):
    '''
    Hinge Loss
    '''
    def __init__(self, alpha: float=1):
        self.alpha = 1
    
    def _assert_shape(self, y: np.ndarray, yhat: np.ndarray):
        assert len(y.shape) == 1, 'y must be 1D matrix'
        assert y.shape == yhat.shape, 'the ground truth and prediction matrices must be of the same size'

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Forward
        :param y: ground truth
            shape = (m,)
        :param yhat: prediction
            shape = (m, 1) or (m,)
        :return hinge loss
            shape = (m,)
        '''
        y = y.reshape(-1)
        yhat = yhat.reshape(-1)
        self._assert_shape(y, yhat)
        return np.maximum(0, self.alpha - y * yhat)

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        '''
        Backward
        :param y: ground truth
            shape = (m,)
        :param yhat: prediction
            shape = (m,)
        :return hinge loss grad
            shape = (m, 1)
        '''
        y = y.reshape(-1)
        yhat = yhat.reshape(-1)
        self._assert_shape(y, yhat)
        mask = np.zeros_like(yhat)
        mask[np.where(y * yhat <= self.alpha)] = 1
        return (-y * mask).reshape(-1, 1)


        