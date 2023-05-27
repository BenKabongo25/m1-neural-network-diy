# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Optimizers

import numpy as np
import module
import loss
import enums


class Optim(object):

    def __init__(self, net: module.Module, loss: loss.Loss, eps: float):
        '''
        :param net: 
            A neural network implemented as a subclass of module.Module.
        :param loss: 
            A loss function implemented as a subclass of loss.Loss.
        :param eps: 
            A learning rate used to scale the gradient before it is used to update the model parameters.
        '''
        self.net = net
        self.loss = loss
        self.eps = eps


    def step(self, X_batch: np.ndarray, y_batch: np.ndarray,
        eval_fn=None, X_valid: np.ndarray=None, y_valid: np.ndarray=None) -> float:
        '''
        Optimizer step

        :param X_batch (numpy.ndarray): 
            The input data used to train the model.
        :param y_batch (numpy.ndarray):
            The target values corresponding to the input data.
        :param eval_fn (callable): 
            A function that calculates the evaluation metric(s) of interest (e.g. accuracy) for the model. 
            If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
        :param X_valid (numpy.ndarray): 
            The validation input data used to evaluate the model during training. 
            If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
        :param y_valid (numpy.ndarray): 
            The target values corresponding to the validation input data. 
            If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
        '''
        # Forward pass
        y_pred = self.net(X_batch)
        loss = self.loss(y_batch, y_pred).mean()

        # Set gradient to 0
        self.net.zero_grad()

        # Backward pass
        delta = self.loss.backward(y_batch, y_pred)
        self.net.backward_update_gradient(X_batch, delta)

        # Update parameters
        self.net.update_parameters(self.eps)

        loss_valid = None
        acc_valid = None

        if X_valid is not None and y_valid is not None:
            y_pred = self.net(X_valid)
            loss_valid = self.loss(y_valid, y_pred).mean()
            if eval_fn is not None:
                acc_valid = eval_fn(self.net, X_valid, y_valid)

        return loss, loss_valid, acc_valid


def SGD(net: module.Module, 
        loss: loss.Loss,
        eps: float,
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int=1_000,
        gradient_descent_mode: int=enums.GradientDescentMode.BATCH,
        batch_size: int=64,
        eval_fn=None,
        X_valid: np.ndarray=None,
        y_valid: np.ndarray=None,
        verbose: bool=True,
        verbose_every: int=10
        ):
    '''
    Performs Stochastic Gradient Descent (SGD) optimization algorithm to train a neural network model.

    :param net (module.Module): 
        A neural network model.
    :param loss (loss.Loss): 
        A loss function to calculate the difference between the predicted values and the actual values.
    :param eps (float): 
        The learning rate used to update the weights of the model.
    :param X (numpy.ndarray): 
        The input data used to train the model.
    :param y (numpy.ndarray):
        The target values corresponding to the input data.
    :param epochs (int): 
        The number of times the training loop should iterate through the entire dataset.
    :param gradient_descent_mode (int): 
        The mode of gradient descent used to update the weights. 
        Can be one of :
            - enums.GradientDescentMode.BATCH, 
            - enums.GradientDescentMode.MINI_BATCH, 
            - enums.GradientDescentMode.STOCHASTIC.
    :param batch_size (int): 
        The number of samples used in each batch during mini-batch gradient descent. 
        Only applicable when gradient_descent_mode is enums.GradientDescentMode.MINI_BATCH.
    :param eval_fn (callable): 
        A function that calculates the evaluation metric(s) of interest (e.g. accuracy) for the model. 
        If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
    :param X_valid (numpy.ndarray): 
        The validation input data used to evaluate the model during training. 
        If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
    :param y_valid (numpy.ndarray): 
        The target values corresponding to the validation input data. 
        If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
    :param verbose (bool): 
        If True, the function prints the loss and evaluation metric(s) at the end of each epoch.
    '''
    
    all_loss = []
    all_loss_valid = []
    all_acc_valid = []

    best_loss_valid = .0
    best_acc_valid = .0
    best_parameters = None

    N = len(X)
    if gradient_descent_mode == enums.GradientDescentMode.STOCHASTIC: 
        batch_size = 1

    optim = Optim(net, loss, eps)

    def batch(X_batch, y_batch, verbose):
        nonlocal best_loss_valid, best_acc_valid, best_parameters

        loss, loss_valid, acc_valid = optim.step(X_batch, y_batch, eval_fn, X_valid, y_valid)

        if verbose: 
            if loss_valid is None:
                print('loss:', loss)
            else:
                if acc_valid is None:
                    print('train loss:', loss, 'val loss:', loss_valid)
                else:
                    print('train loss:', loss, 'val loss:', loss_valid, 'val eval_fn:', acc_valid)

        all_loss.append(loss)

        if loss_valid is not None: 
            all_loss_valid.append(loss_valid)
            if acc_valid is None and loss_valid < best_loss_valid:
                best_loss_valid = loss_valid
                best_parameters = optim.net.get_parameters()

        if acc_valid is not None: 
            all_acc_valid.append(acc_valid)
            if acc_valid > best_acc_valid:
                best_acc_valid = acc_valid
                best_parameters = optim.net.get_parameters()

    
    if verbose: print('Train : -----------------------------------')
    for i in range(epochs):
        verbose_epoch = verbose and ((i + 1) % (epochs // verbose_every) == 0) if epochs > verbose_every else True
        if verbose_epoch: print(f'Epoch {i + 1}: ', end='')

        if gradient_descent_mode == enums.GradientDescentMode.BATCH:
            batch(X, y, verbose_epoch)
        else:
            idx = np.arange(N)
            np.random.shuffle(idx)
            if verbose_epoch: print()
            for bi in range(0, N, batch_size):
                bi_ = min(N, bi + batch_size)
                batch_idx = idx[bi : bi_]
                if verbose_epoch: print(f'Batch {bi_}: ', end='')
                batch(X[batch_idx], y[batch_idx], verbose_epoch)

    if verbose: print('-------------------------------------------')
    return all_loss, all_loss_valid, all_acc_valid, best_parameters
