import theano
from theano import tensor as T
import numpy as np
import backend as K
import optimizer
import regularizers
import layers
import net
import time
import math
import cPickle as pkl
import sys
from scipy import io
sys.setrecursionlimit(2000)
import random
from SplitDataIntoArrays import indexContent
import DataHelper
import evaluation

class Model():

    def __init__(self, nets=[]):
        self.nets = []
        for net in nets:
            self.add(net)

    def add(self, net):
        self.nets.append(net)

    def connect(self,net_pre, net_aft):
        net_aft.layers[0].set_previous(net_pre.layers[-1])

    def connect_hierarchical(self,net_first_layer,net_second_layer,model_options):
        # convert the hidden content to higher layer LSTM unit
        for i in xrange(model_options['v_length']/model_options['hiera_step']):
            net_second_layer.layers[2*i].set_previous(net_first_layer.layers[(i+1)*model_options['hiera_step']-1])

    @property
    def updates(self):
        updates = []
        for net in self.nets:
            if hasattr(net,'updates'):
                updates += net.updates
        return updates

    @property
    def params(self):
        params = []
        for net in self.nets:
            if hasattr(net,'params'):
                params += net.params
        return params

    @property
    def regularizers(self):
        regularizers = []
        for net in self.nets:
            regularizers += net.regularizers
        return regularizers

    def get_out(self,train=True):
        outs = []
        for net_i in xrange(len(self.nets)):
            outs.append(self.nets[net_i].get_out(train=train))
        return outs[-1]
    def init_state(self):
        self.init_h = T.matrix(name='input_hidden', dtype='float32')
        self.init_m = T.matrix(name='input_memory', dtype='float32')

        self.init_h_high = T.matrix(name='input_h_hiera', dtype='float32')
        self.init_m_high = T.matrix(name='input_m_hiera', dtype='float32')

        self.init_h_ru = T.matrix(name='init_h_ru', dtype='float32')
        self.init_m_ru = T.matrix(name='init_m_ru', dtype='float32')

        self.init_h_rv = T.matrix(name='init_h_rv', dtype='float32')
        self.init_m_rv = T.matrix(name='init_m_rv', dtype='float32')

        self.init_h_rm = T.matrix(name='init_h_rm', dtype='float32')
        self.init_m_rm = T.matrix(name='init_h_rm', dtype='float32')

    def compile(self, options):
        '''Configure the learning process.
        '''
        # input of model
        self.X = T.tensor3(name='input_frames', dtype='float32')
        self.H = T.matrix(name='H', dtype='float32')
        self.idx = T.vector(name='idx',dtype='int32')

        netlu = self.nets[0]
        netru = self.nets[1]
        netrv = self.nets[2]
        netrm = self.nets[3]
        net_hiera = self.nets[4]
        net_ru_high = self.nets[5]
        net_rv_high = self.nets[6]
        net_rm_high = self.nets[7]

        self.init_state()

        netlu.set_input([self.init_h,self.init_m])
        net_hiera.set_input([self.init_h_high,self.init_m_high])
        netru.set_input([self.init_h_ru, self.init_m_ru])
        netrv.set_input([self.init_h_rv, self.init_m_rv])
        netrm.set_input([self.init_h_rm, self.init_m_rm])

        #set the image feature as input
        idx = 0
        for l in netlu.layers:
            if hasattr(l, 'has_input_frame'):
                if l.has_input_frame:
                    l.input_frame = self.X[:,idx,:]
                    idx += 1
        assert  idx == options['v_length']
        print "start loading SS matrix..."
        time1 = time.time()
        print options['SS_path']
        SS = pkl.load(open(options['SS_path'])).astype(np.int8)
        print 'SS.shape: ',SS.shape
        # for debug
        # SS = np.zeros((train_data.data_size_,train_data.data_size_)).astype(np.float32)
        # SS[:,1] = np.ones((1,train_data.data_size_))

        SS_shared = theano.shared(value=SS, name='SS_shared')
        time2 = time.time()
        print "load SS matrix costs: ", time2 - time1

        #H_shared = theano.shared(value=T.zeros(shape=[]))

        def comp_(train):
            netlu.set_out(train=train)
            #nethiera.set_out(train=train)
            idx_hie = 0
            for i in net_hiera.layers:
                if hasattr(i,'has_input_frame'):
                    if i.has_input_frame:
                        i.input_frame = netlu.layers[(idx_hie+1)*options['hiera_step']-1].get_output(train=train)[0]
                        idx_hie += 1
            net_hiera.set_out(train=train)

            net_ru_high.set_out(train=train)
            idx_ru = 0
            for i in netru.layers:
                if hasattr(i,'has_input_frame'):
                    if i.has_input_frame:
                        if netru.layers.index(i)%options['hiera_step']==0:
                            i.input_frame = net_ru_high.layers[idx_ru].get_output(train=train)[0]
                            idx_ru += 1
                        else:
                            i.input_frame = net_ru_high.layers[idx_ru-1].get_output(train=train)[0]

            net_rv_high.set_out(train=train)
            idx_rv = 0
            for i in netrv.layers:
                if hasattr(i, 'has_input_frame'):
                    if i.has_input_frame:
                        if netrv.layers.index(i)%options['hiera_step']==0:
                            i.input_frame = net_rv_high.layers[idx_rv].get_output(train=train)[0]
                            idx_rv += 1
                        else:
                            i.input_frame = net_ru_high.layers[idx_rv-1].get_output(train=train)[0]

            net_rm_high.set_out(train=train)
            netrm.layers[0].input_frame= net_rm_high.layers[0].get_output(train=train)[0]

            if not train:
                [my_H, my_M] = net_hiera.get_out_idx(-2)
                print 'compile encoder...'
                self._encoder = theano.function([self.X,
                                                 self.init_h,self.init_m,
                                                 self.init_h_high,self.init_m_high], my_H)
            #construct pairwise loss
            lamb = options['lamb']
            [my_H, my_M] = net_hiera.get_out_idx(-2)
            #get binary code from network

            my_B = T.sgn(my_H)

            #self.H_: batch_size * nbits
            self.H_ = T.set_subtensor(self.H[self.idx,:],my_H[:self.idx.shape[0],:])  #add the hidden state into H

            #self.SS_ = self.SS[self.idx]
            if self.idx.shape[0] == options['batch_size']:
                self.SS_ = SS_shared[self.idx]    # SS_: batch_size * train_size
            else:
                # SS_: batch_size * train_size
                self.SS_ = T.set_subtensor(T.zeros((options['batch_size'],SS_shared.shape[1]))[:self.idx.shape[0]],SS_shared[self.idx])


            loss_pairwise = T.sum(T.square(T.dot(my_H,self.H_.transpose())/options['dim_proj']-self.SS_))

            loss_pairwise +=lamb*(T.sum(T.square(my_H-my_B)))

            self.y_pred = netru.get_out(train=train)
            assert len(self.y_pred) == options['v_length']
            loss_backward = T.sum(T.sqr(self.X[:,-1,:] - self.y_pred[0]))
            for i in xrange(1,options['v_length']):
                loss_backward += T.sum(T.sqr(self.X[:,-1-i,:] - self.y_pred[i]))

            self.y_pred2 = netrv.get_out(train=train)
            assert len(self.y_pred2) == options['v_length']
            loss_forward = T.sum(T.sqr(self.X[:,0,:] - self.y_pred2[0]))
            for i in xrange(1,options['v_length']):
                loss_forward += T.sum(T.sqr(self.X[:,i,:] - self.y_pred2[i]))
            
            self.y_mean = netrm.get_out(train=train)
            assert len(self.y_mean) == 1
            loss_mean= options['v_length'] * T.sum(T.sqr(T.mean(self.X, axis=1) - self.y_mean[0]))
            whts = options['weights']

            reconstruction_loss = whts[0]*loss_backward + whts[1]*loss_forward + whts[2]*loss_mean
            #add pairwise loss
            loss = loss_pairwise*options['pairwise_weight']+reconstruction_loss*(1-options['pairwise_weight'])
            #loss = loss_pairwise + reconstruction_loss

            for r in self.regularizers:
                loss = r(loss)
            
            if train:
                self.optimizer = eval('optimizer.'+ options['optimizer'])(self.params, lr=options['lrate'])
                updates = self.optimizer.get_updates(self.params, loss)
                updates += self.updates
                print 'compile train...'
                start_time = time.time()
                self._train = theano.function([self.X, self.idx, self.H,
                                               self.init_h, self.init_m,
                                               self.init_h_high, self.init_m_high,
                                               self.init_h_ru, self.init_m_ru,
                                               self.init_h_rv, self.init_m_rv,
                                               self.init_h_rm, self.init_m_rm],
                                              [self.H_ ,loss, loss_pairwise,reconstruction_loss], updates=updates)
                end_time = time.time()
                print 'spent %f seconds'  % (end_time-start_time)
            else:
                print 'compile test...'
                start_time = time.time()
                self._test = theano.function([self.X,self.idx,self.H,
                                              self.init_h, self.init_m,
                                              self.init_h_high, self.init_m_high,
                                              self.init_h_ru, self.init_m_ru,
                                              self.init_h_rv, self.init_m_rv,
                                              self.init_h_rm, self.init_m_rm], loss)
                end_time = time.time()
                print 'spent %f seconds'  % (end_time-start_time)

        comp_(train=True)
        comp_(train=False)
        print "Compile Done!"



    def train(self, train_data_path, test_data, options):

        validFreq = options['validFreq']
        saveFreq = options['saveFreq']
        dispFreq = options['dispFreq']
        max_iter = options['max_iter']
        saveto =options['saveto']

        train_loss_his = []
        test_loss_his = []

        start_time = time.time()

        #test_loss_ = self.test_loss(self._test, test_data, options)
        # test_loss_his.append(test_loss_)
        # print 'Valid cost:', test_loss_

        train_loss = 0.
        records_file = open(options['record_path'],'w+')
        file_name = options['train_data_file_path'] + 'fcv_train_feats.h5'
        train_data = DataHelper.DataHelper(options['v_length'], options['batch_size'],
                                           options['dim_frame'],
                                           data_file=file_name, train=True)
        H = np.zeros([train_data.data_size_, options['dim_proj']],dtype=np.float32)

        try:
            for uidx in xrange(1,max_iter+1):
                #get splits of an epoch
                for eidx in xrange(1,options['train_splits_num']+1):
                    #for YFCC
                    #file_name = options['train_data_file_path']+'yfcc_train_feats_'+str(eidx)+'.h5'
                    #for FCV
                    file_name = options['train_data_file_path'] + 'fcv_train_feats.h5'

                    train_data = DataHelper.DataHelper(options['v_length'], options['batch_size'],
                                                       options['dim_frame'],
                                                        data_file= file_name, train=True)
                    print 'loading data:'+file_name
                    #get the batch train data

                    m = train_data.data_size_/train_data.batch_size_
                    if  train_data.data_size_%train_data.batch_size_ == 0:
                        m = m
                    else:
                        m += 1
                    print 'm: ',m
                    for i in range(0,m):
                        #if i % 10 ==0:
                            #print i
                        if i == (m-1):
                            x = indexContent(train_data,train_data.idx_[i*options['batch_size']:])
                            idxs = train_data.idx_[i*options['batch_size']:]

                        else:
                            x = indexContent(train_data,train_data.idx_[i*options['batch_size']:(i+1)*options['batch_size']])
                            idxs = train_data.idx_[i*options['batch_size']:(i+1)*options['batch_size']]

                        [H, train_loss, loss_pairwise,reconstruction_loss] = self._train(
                            x,idxs,H,
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                            np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32))
                        if i % 10 == 0:
                            print 'Epoch: ',uidx,'\tPart: ',eidx,'\tBatch: ',i,'\tCost: ',train_loss,'\tpairwise_loss: ',loss_pairwise,'\trestruction_loss: ',reconstruction_loss
                            records_file.write('Epoch: '+str(uidx)+'\tPart: '+str(eidx)+'\tBatch: '+str(i)+'\tCost: '+str(train_loss)+'\tpairwise_loss: '+str(loss_pairwise)+'\trestruction_loss'+str(reconstruction_loss)+'\n')

                if uidx%options['validFreq'] == 0:
                    print 'start testing...'
                    maps = evaluation.test(self._encoder,options,uidx)
                if np.isnan(train_loss) or np.isinf(train_loss):
                    print 'bad cost detected: ', train_loss

                if np.mod(uidx, dispFreq) == 0 or uidx is 1:
                    train_loss = train_loss/(x.shape[0]*x.shape[1])
                    train_loss_his.append(train_loss)
                    print 'Step ', uidx,  'Train cost:', train_loss

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print 'Saving...',
                    params_to_save = self.get_params_value()
                    updates_value = self.get_updates_value()
                    np.savez(saveto, params=params_to_save, updates_v=updates_value,
                             train_loss_his=train_loss_his)
                    pkl.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Save Done'


        except KeyboardInterrupt:
            print "Training interupted"
            print 'Saving records!'
            records_file.close()

        if saveto:
            print 'Saving...',
            params_to_save = self.get_params_value()
            updates_value = self.get_updates_value()
            np.savez(saveto, params=params_to_save, updates_v=updates_value,
                     train_loss_his=train_loss_his, test_loss_his=test_loss_his)
            pkl.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
            print 'Save Done'

        end_time = time.time()
        print  ('Training took %.1fs' % (end_time - start_time))


    def test(self, test_data, options):

        max_iter = int(math.ceil(float(test_data.data_size_)/float(test_data.batch_size_)))
        start_time = time.time()

        test_loss_ = self.test_loss(self._test, test_data, options)
        print 'Valid cost:', test_loss_

        try:
            for uidx in xrange(1,max_iter+1):
                x = test_data.GetBatch()
                hidden_orig_cpu = self._encoder(x,
                    np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                    np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                    np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                    np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32))
                io.savemat('hidden_' + str(uidx) + '.mat', {'hidden': hidden_orig_cpu})

        except KeyboardInterrupt:
            print "Test interupted"

        end_time = time.time()
        print  ('Testing took %.1fs' % (end_time - start_time))


    def test_loss(self, f_pred, data, options):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        """
        pred_loss = 0.
        sum_iter = int(math.ceil(float(data.data_size_)/float(data.batch_size_)))
        for i in range(sum_iter):
            x = data.GetBatch()

            pred_loss += f_pred(x,data.
                                np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                                np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                                np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32),
                                np.zeros((x.shape[0], options['dim_proj']), dtype=np.float32))
        pred_loss = pred_loss/(data.data_size_*data.seq_length_)
        return pred_loss

    def get_params_value(self):
        new_params = [par.get_value() for par in self.params]
        return new_params

    def get_updates_value(self):
        updates = [par.get_value() for (par,up) in self.updates]
        return  updates

    def reload_params(self, params_file):
        print 'Reloading model params'
        ff = np.load(params_file)
        new_parms = ff['params']
        for idx, par in enumerate(self.params):
            K.set_value(par, new_parms[idx])
        new_updates = ff['updates_v']
        for idx, (par,up) in enumerate(self.updates):
            K.set_value(par, new_updates[idx])
