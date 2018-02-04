import theano
from theano import tensor as T
from theano import config
import numpy as np
import backend as K
import regularizers
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
import initializations
import activations


class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'input_shape',
                          'name',
                          'output_shape'}
        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'Keyword argument not understood: ' + kwarg
        if 'input_shape' in kwargs:
            self.input_shape = kwargs['input_shape']
        self.name = self.__class__.__name__.lower()
        if 'name' in kwargs:
            self.name = kwargs['name']
        if not hasattr(self, 'params'):
            self.params = []

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def output_shape(self):
        # default assumption: tensor shape unchanged.
        return self.input_shape

    def set_previous(self, layer):
        '''Connect a layer to its parent in the computational graph.
        '''
        self.previous = layer

    def build(self):
        '''Instantiation of layer weights.
        Called after `set_previous`, or after `set_input_shape`,
        once the layer has a defined input shape.
        Must be implemented on all layers that have weights.
        '''
        return 'build have not been implemented'

    def set_output(self,train=False):
        # default assumption: tensor unchanged.
        return 'set_output have not been implemented'

    def get_output(self,train=False):
        if not hasattr(self, 'output'):
            self.set_output(train)
        return self.output

    def get_input(self,train=False):
        if hasattr(self, 'previous'):        
            previous_output = self.previous.get_output(train=train)           
            return previous_output
        elif hasattr(self, 'input'):
            return self.input
        else:
            raise Exception('Layer is not connected and is not an input layer.')

    def get_params(self):
        if hasattr(self, 'params'):
            params = self.params
        else:
            params = []

        return params

    def get_regularizers(self):

        if hasattr(self, 'regularizers'):
            regularizers = self.regularizers
        else:
            regularizers = []

        return regularizers

    def get_updates(self):

        if hasattr(self, 'bn'):
            updates = self.bn.updates
            #print 'has updates',len(updates)
        else:
            #print 'no updates'
            updates = []

        return updates

f_init = initializations.get('glorot_uniform')
f_init2 = initializations.get('uniform2')
f_inner_init = initializations.get('orthogonal')
f_forget_bias_init = initializations.get('one')


class LstmParams(object):
    def __init__(self, num_lstm, dim_frame,output_dim):
        self.num_lstm = num_lstm
        self.dim_frame = dim_frame
        self.output_dim = output_dim

        self.W_in_to_ingate = f_init((self.dim_frame, self.num_lstm))
        self.W_in_to_forgetgate = f_init((self.dim_frame, self.num_lstm))
        self.W_in_to_cell = f_init((self.dim_frame, self.num_lstm))
        self.W_in_to_outgate = f_init((self.dim_frame, self.num_lstm))

        self.W_hid_to_ingate = f_inner_init((self.num_lstm, self.num_lstm))
        self.W_hid_to_forgetgate = f_inner_init((self.num_lstm, self.num_lstm))
        self.W_hid_to_cell = f_inner_init((self.num_lstm, self.num_lstm))
        self.W_hid_to_outgate = f_inner_init((self.num_lstm, self.num_lstm))

        self.b_ingate = K.zeros((self.num_lstm,))
        self.b_forgetgate = f_forget_bias_init((self.num_lstm,))
        self.b_cell = K.zeros((self.num_lstm,))
        self.b_outgate = K.zeros((self.num_lstm,))

        self.W_cell_to_ingate = f_init2((self.num_lstm,))
        self.W_cell_to_forgetgate = f_init2((self.num_lstm,))
        self.W_cell_to_outgate = f_init2((self.num_lstm,))

        self.W_output = f_init((self.num_lstm,self.output_dim))
        self.b_output = K.zeros((self.output_dim,))


class LSTM_Unit(Layer):
    def __init__(self, lid, par, batch_size, num_lstm, dim_frame,output_dim, l2_decay, activation='tanh', inner_activation='sigmoid', **kwargs):
        self.num_lstm = num_lstm
        self.dim_frame = dim_frame
        self.output_dim = output_dim
        self.has_input_frame = True
        self.batch_size = batch_size
        self.inner_activation = activations.get(inner_activation)
        self.activation = activations.get(activation)
        self.bn = BatchNormalization(lid, self.batch_size, self.num_lstm)
        self.bn.build()
        self.lstmpar = par
        self.l2_decay = l2_decay
        self.id = lid
        kwargs['input_shape'] = (self.batch_size,self.num_lstm)
        super(LSTM_Unit, self).__init__(**kwargs)

    def build(self):

        self.params = [self.bn.gamma, self.bn.beta]

        if self.id == 0:
            #self.input = [K.zeros((self.batch_size,self.num_lstm)), K.zeros((self.batch_size,self.num_lstm))]

            params = [self.lstmpar.W_in_to_ingate, self.lstmpar.W_in_to_forgetgate, self.lstmpar.W_in_to_cell, self.lstmpar.W_in_to_outgate,
                        self.lstmpar.W_hid_to_ingate, self.lstmpar.W_hid_to_forgetgate, self.lstmpar.W_hid_to_cell, self.lstmpar.W_hid_to_outgate,
                        self.lstmpar.b_ingate, self.lstmpar.b_forgetgate, self.lstmpar.b_cell, self.lstmpar.b_outgate, self.lstmpar.W_cell_to_ingate,
                        self.lstmpar.W_cell_to_forgetgate, self.lstmpar.W_cell_to_outgate,self.lstmpar.W_output,self.lstmpar.b_output]

            self.params = params + self.params

        self.regularizers = []
        for par in self.params:
            regularizer = regularizers.WeightRegularizer(l1=0., l2=self.l2_decay)
            regularizer.set_param(par)
            self.regularizers.append(regularizer)

    def step(self, input_n, cell_previous, hid_previous, train):
        input_to_in = T.dot(input_n, self.lstmpar.W_in_to_ingate) + K.reshape(self.lstmpar.b_ingate,[1,self.num_lstm])
        input_to_forget = T.dot(input_n, self.lstmpar.W_in_to_forgetgate) + K.reshape(self.lstmpar.b_forgetgate,[1,self.num_lstm])
        input_to_cell = T.dot(input_n, self.lstmpar.W_in_to_cell) + K.reshape(self.lstmpar.b_cell,[1,self.num_lstm])
        input_to_out = T.dot(input_n, self.lstmpar.W_in_to_outgate) + K.reshape(self.lstmpar.b_outgate,[1,self.num_lstm])


        ingate = input_to_in + T.dot(hid_previous, self.lstmpar.W_hid_to_ingate)
        forgetgate = input_to_forget + T.dot(hid_previous, self.lstmpar.W_hid_to_forgetgate)
        cell_input = input_to_cell + T.dot(hid_previous, self.lstmpar.W_hid_to_cell)
        outgate = input_to_out + T.dot(hid_previous, self.lstmpar.W_hid_to_outgate)


        # Compute peephole connections
        ingate += cell_previous * K.reshape(self.lstmpar.W_cell_to_ingate,[1,self.num_lstm])
        forgetgate += cell_previous * K.reshape(self.lstmpar.W_cell_to_forgetgate,[1,self.num_lstm])

        # Apply nonlinearities
        ingate = K.sigmoid(ingate)
        forgetgate = K.sigmoid(forgetgate)
        cell_input = K.tanh(cell_input)

        # Compute new cell value
        cell = forgetgate * cell_previous + ingate * cell_input
        cell_bn = self.bn.set_output(cell,train=train)

        outgate += cell_bn * K.reshape(self.lstmpar.W_cell_to_outgate,[1,self.num_lstm])
        outgate = K.sigmoid(outgate)

        # Compute new hidden unit activation    
        hid = outgate * cell_bn
        return [cell_bn, hid]

    def set_output(self,train=False):
        [X_H, X_M] = self.get_input(train=train)
        assert hasattr(self, 'input_frame')
        [cell, hid] = self.step(self.input_frame, X_M, X_H, train)
        self.output = [hid, cell]
        self.output_frame = T.dot(hid, self.lstmpar.W_output) + K.reshape(self.lstmpar.b_output, [1, self.output_dim])



class BatchNormalization(object):

    def __init__(self, lid, bacth_size, num_lstm, epsilon=1e-10, axis=1, momentum=0.99, **kwargs):
        self.init = initializations.get("one")
        self.epsilon = epsilon
        self.batch_size = bacth_size
        self.axis = axis
        self.momentum = momentum
        self.num_lstm = num_lstm
        self.id = lid

    def build(self):
        self.gamma = K.ones((self.num_lstm,))
        self.beta = K.zeros((self.num_lstm,))
        
        self.running_mean = K.zeros((self.num_lstm,))
        self.running_std = K.ones((self.num_lstm,))
        self.updates = [(self.running_mean, None), (self.running_std, None)]

    def set_output(self, X, train=False):

        input_shape = (self.batch_size, self.num_lstm)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        if train:
            m = K.mean(X, axis=reduction_axes)
            brodcast_m = K.reshape(m, broadcast_shape)
            std = K.mean(K.square(X - brodcast_m) + self.epsilon, axis=reduction_axes)
            std = K.sqrt(std)
            brodcast_std = K.reshape(std, broadcast_shape)
            mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
            std_update = self.momentum * self.running_std + (1-self.momentum) * std
            self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]
            X_normed = (X - brodcast_m) / (brodcast_std + self.epsilon)
        else:
            brodcast_m = K.reshape(self.running_mean, broadcast_shape)
            brodcast_std = K.reshape(self.running_std, broadcast_shape)
            X_normed = ((X - brodcast_m) /
                            (brodcast_std + self.epsilon))
        out = K.reshape(self.gamma, broadcast_shape) * X_normed + K.reshape(self.beta, broadcast_shape)

        return out

class Round3(UnaryScalarOp):

    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,

round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

class BinaryLayer(Layer):
    def __init__(self, lid, **kwargs):
        self.activation = binary_tanh_unit
        self.id = lid
        super(BinaryLayer, self).__init__(**kwargs)
    def build(self):
        pass
    def set_output(self,train=False):
        [X_H, X_M] = self.get_input(train=train)
        X_H_b = self.activation(X_H)
        self.output = [X_H_b, X_M]


class DecLstmParams(object):
    def __init__(self, num_lstm, dim_frame):
        self.num_lstm = num_lstm
        self.dim_frame = dim_frame

        self.W_hid_to_ingate = f_inner_init((self.num_lstm, self.num_lstm))
        self.W_hid_to_forgetgate = f_inner_init((self.num_lstm, self.num_lstm))
        self.W_hid_to_cell = f_inner_init((self.num_lstm, self.num_lstm))
        self.W_hid_to_outgate = f_inner_init((self.num_lstm, self.num_lstm))

        self.b_ingate = K.zeros((self.num_lstm,))
        self.b_forgetgate = f_forget_bias_init((self.num_lstm,))
        self.b_cell = K.zeros((self.num_lstm,))
        self.b_outgate = K.zeros((self.num_lstm,))

        self.W_cell_to_ingate = f_init2((self.num_lstm,))
        self.W_cell_to_forgetgate = f_init2((self.num_lstm,))
        self.W_cell_to_outgate = f_init2((self.num_lstm,))

        self.W_output = f_init((self.num_lstm, self.dim_frame))
        self.b_output = K.zeros((self.dim_frame,))


class LSTM_Dec(Layer):
    def __init__(self, lid, par, bacth_size, num_lstm, dim_frame, l2_decay, use_th=True, activation='tanh', inner_activation='sigmoid', **kwargs):

        self.num_lstm = num_lstm
        self.dim_frame = dim_frame
        self.batch_size = bacth_size
        self.inner_activation = activations.get(inner_activation)
        self.activation = activations.get(activation)
        self.lstmpar = par
        self.l2_decay = l2_decay
        self.use_th = use_th
        self.bn = BatchNormalization(lid, self.batch_size, self.num_lstm)
        self.bn.build()
        self.id = lid
        kwargs['input_shape'] = (self.batch_size,self.num_lstm)
        super(LSTM_Dec, self).__init__(**kwargs)

    def build(self):

        self.params = [self.bn.gamma, self.bn.beta]

        if self.id == 0:

            params = \
                [self.lstmpar.W_hid_to_ingate, self.lstmpar.W_hid_to_forgetgate, self.lstmpar.W_hid_to_cell, self.lstmpar.W_hid_to_outgate,
                self.lstmpar.b_ingate, self.lstmpar.b_forgetgate, self.lstmpar.b_cell, self.lstmpar.b_outgate, self.lstmpar.W_cell_to_ingate,
                self.lstmpar.W_cell_to_forgetgate, self.lstmpar.W_cell_to_outgate, self.lstmpar.W_output, self.lstmpar.b_output]

            self.params = params + self.params

        self.regularizers = []
        for par in self.params:
            regularizer = regularizers.WeightRegularizer(l1=0., l2=self.l2_decay)
            regularizer.set_param(par)
            self.regularizers.append(regularizer)


    def step(self, cell_previous, hid_previous, train):

        ingate = T.dot(hid_previous, self.lstmpar.W_hid_to_ingate) + K.reshape(self.lstmpar.b_ingate,[1,self.num_lstm])
        forgetgate = T.dot(hid_previous, self.lstmpar.W_hid_to_forgetgate) + K.reshape(self.lstmpar.b_forgetgate,[1,self.num_lstm])
        cell_input = T.dot(hid_previous, self.lstmpar.W_hid_to_cell) + K.reshape(self.lstmpar.b_cell,[1,self.num_lstm])
        outgate = T.dot(hid_previous, self.lstmpar.W_hid_to_outgate) + K.reshape(self.lstmpar.b_outgate,[1,self.num_lstm])

        # Compute peephole connections
        ingate += cell_previous * K.reshape(self.lstmpar.W_cell_to_ingate,[1,self.num_lstm])
        forgetgate += cell_previous * K.reshape(self.lstmpar.W_cell_to_forgetgate,[1,self.num_lstm])

        # Apply nonlinearities
        ingate = K.sigmoid(ingate)
        forgetgate = K.sigmoid(forgetgate)
        cell_input = K.tanh(cell_input)

        # Compute new cell value
        cell = forgetgate * cell_previous + ingate * cell_input
        cell_bn = self.bn.set_output(cell,train=train)

        outgate += cell_bn * K.reshape(self.lstmpar.W_cell_to_outgate,[1,self.num_lstm])
        outgate = K.sigmoid(outgate)

        # Compute new hidden unit activation
        if self.use_th:
            hid = outgate * K.tanh(cell_bn)
        else:
            hid = outgate * cell_bn
        return [cell_bn, hid]

    def set_output(self,train=False):
        [X_H, X_M] = self.get_input(train=train)
        [cell, hid] = self.step(X_M, X_H, train)
        self.output = [hid, cell]
        self.output_frame = T.dot(hid, self.lstmpar.W_output) + K.reshape(self.lstmpar.b_output,[1,self.dim_frame])