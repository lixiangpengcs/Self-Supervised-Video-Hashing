import theano
from theano import tensor as T
from theano import config
import numpy as np
import backend as K

class Optimizer(object):
    '''Abstract optimizer base class.
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []

    def get_updates(self, params, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = T.grad(loss, params)
        return grads


class Adam(Optimizer):
    '''Adam optimizer.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, lda = 1-1e-8, epsilon=1e-8,
                 *args, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.lda = K.variable(lda)
        self.epsilon = K.variable(epsilon)
        self.m = []
        self.v = []
        for par in params:
            m = K.variable(np.zeros(K.get_value(par).shape))
            v = K.variable(np.zeros(K.get_value(par).shape))
            self.m += [m]
            self.v += [v]

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1.)]

        t = self.iterations + 1
        beta_2t = K.sqrt(1 - K.pow(self.beta_2, t))
        lr_t = self.lr * beta_2t / (1 - K.pow(self.beta_1, t))

        for p, g, m, v in zip(params, grads, self.m, self.v):

            beta_1t = self.beta_1 * K.pow(self.lda, t-1)
            m_t = (beta_1t * m) + (1 - beta_1t) * g
            v_t = (self.beta_2 * v) + (1 - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon * beta_2t)

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))
            self.updates.append((p, p_t))
        return self.updates

class SGD(Optimizer):
    '''Stochastic gradient descent, with support for momentum,
    decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    '''
    def __init__(self, params, lr=0.001, momentum=0.9, decay=0.9, nesterov=False,
                 *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)
        self.lr_decay_after = K.variable(10000.)
        self.m = []
        for par in params:
            m = K.variable(np.zeros(K.get_value(par).shape))
            self.m.append(m)

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (self.decay ** (self.iterations / self.lr_decay_after))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, m in zip(params, grads, self.m):
            #m = K.variable(np.zeros(K.get_value(p).shape))  # momentum
            v = self.momentum * m - lr * g  # velocity
            v = T.clip(v, -0.0001, 0.0001)
            self.updates.append((m, v))
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append((p, new_p))  # apply constraints
        return self.updates