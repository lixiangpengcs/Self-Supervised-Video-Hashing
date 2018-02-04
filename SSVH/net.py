import theano
from theano import tensor as T
import numpy as np
import backend as K
import optimizer
import regularizers
import layers
import time
import sys

class Net():

    def __init__(self, layers=[]):
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self,layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        layer.build()

    @property
    def updates(self):
        updates = []
        for l in self.layers:
            updates += l.get_updates()
        return updates

    @property
    def params(self):
        params = []
        for l in self.layers:
            if hasattr(l,'params'):
                params += l.get_params()
        return params

    @property
    def regularizers(self):
        regularizers = []
        for l in self.layers:
            regularizers += l.get_regularizers()
        return regularizers

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def get_output(self, train=False):
        return self.layers[-1].get_output(train)

    def get_input(self):
        assert hasattr(self.layers[0], 'input'),'Net has no input'
        return self.layers[0].get_input()

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def input(self):
        return self.get_input()

    def get_out(self,train=True):
        outs = []
        for layer_i in xrange(len(self.layers)):
            self.layers[layer_i].set_output(train)
            if hasattr(self.layers[layer_i],'output_frame'):
                outs.append(self.layers[layer_i].output_frame)
        return outs

    def get_out_idx(self, idx):
        '''
        for layer_i in xrange(len(self.layers)):
            self.layers[layer_i].set_output(train)
        '''
        return self.layers[idx].output

    def set_out(self,train=True):
        #set the LSTM units into a complete layers,which the previous unit get the output of the previous unit
        for layer_i in xrange(len(self.layers)):
            self.layers[layer_i].set_output(train)

    def set_input(self, inputs):
        self.layers[0].input = inputs
