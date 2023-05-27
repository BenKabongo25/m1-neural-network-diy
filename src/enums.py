# Machine Learning
# DAC, Sorbonne University
# Ben Kabongo
#
# Enumerations


class Initialization(object):
    '''
    module weight initialization mode
    zero, one, random, uniform, xavier, lecun
    '''
    ZERO    = 0
    ONE     = 1
    RANDOM  = 2
    UNIFORM = 3
    XAVIER  = 4
    LECUN   = 5


class GradientDescentMode(object):
    '''
    Gradient Descent mode
    batch, mini_batch, stochastic
    '''
    BATCH      = 0
    MINI_BATCH = 1
    STOCHASTIC = 2