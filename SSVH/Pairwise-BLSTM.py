import cPickle as pkl
import sys
import time
import numpy
import logging

from net import *
from model import *
from layers import *
import DataHelper

def build_model(model_options):
    # Encoder LSTMs
    net_lu = Net()
    lstmpar_en = LstmParams(model_options['dim_proj'], model_options['dim_frame'], model_options['output_dim'])
    for i in xrange(model_options['v_length']):
        net_lu.add(
            LSTM_Unit(i, lstmpar_en, model_options['batch_size'], model_options['dim_proj'], model_options['dim_frame'],
                      model_options['output_dim'], model_options['l2_decay']))

    # Encoder LSTMs : the second layer of the encoder LSTMs
    net_en_hiera = Net()
    lstmpar_en_sec = LstmParams(model_options['dim_proj'], model_options['dim_proj'], model_options['output_dim'])
    for i in xrange(model_options['v_length'] / model_options['hiera_step']):
        net_en_hiera.add(LSTM_Unit(2 * i, lstmpar_en_sec, model_options['batch_size'], model_options['dim_proj'],
                                   model_options['dim_proj'], model_options['output_dim'], model_options['l2_decay']))
        net_en_hiera.add(BinaryLayer(2 * i + 1))

    # the higher layer of Backword Decoder LSTMs
    net_ru_high = Net()
    decpar_ru_high = DecLstmParams(model_options['dim_proj'], model_options['dim_proj'])
    for i in xrange(model_options['v_length'] / model_options['hiera_step']):
        net_ru_high.add(LSTM_Dec(i, decpar_ru_high,
                                 model_options['batch_size'], model_options['dim_proj'],
                                 model_options['dim_proj'], model_options['l2_decay']))

    # the lower layer of Backword Decoder LSTMs
    net_ru = Net()
    decpar_ru = LstmParams(model_options['dim_proj'], model_options['dim_proj'], model_options['output_dim'])
    for i in xrange(model_options['v_length']):
        net_ru.add(LSTM_Unit(i, decpar_ru, model_options['batch_size'], model_options['dim_proj'],
                             model_options['dim_proj'], model_options['output_dim'], model_options['l2_decay']))

    # the higher layer of Forword Decoder LSTMs
    net_rv_high = Net()
    decpar_rv_high = DecLstmParams(model_options['dim_proj'], model_options['dim_proj'])
    for i in xrange(model_options['v_length'] / model_options['hiera_step']):
        net_rv_high.add(LSTM_Dec(i, decpar_rv_high,
                                 model_options['batch_size'], model_options['dim_proj'],
                                 model_options['dim_proj'],model_options['l2_decay']))

    # the lower layer of Forword Decoder LSTMs
    net_rv = Net()
    decpar_rv = LstmParams(model_options['dim_proj'], model_options['dim_proj'], model_options['output_dim'])
    for i in xrange(model_options['v_length']):
        net_rv.add(LSTM_Unit(i, decpar_rv, model_options['batch_size'], model_options['dim_proj'],
                             model_options['dim_proj'], model_options['output_dim'], model_options['l2_decay']))

    # Other Decoder LSTMs
    # the higher layer
    net_rm_high = Net()
    decpar_rm_high = DecLstmParams(model_options['dim_proj'], model_options['dim_proj'])
    net_rm_high.add(LSTM_Dec(0, decpar_rm_high, model_options['batch_size'], model_options['dim_proj'],
                             model_options['dim_proj'],model_options['l2_decay']))

    # the lower layer
    net_rm = Net()
    encpar_rm = LstmParams(model_options['dim_proj'], model_options['dim_proj'], model_options['output_dim'])
    net_rm.add(LSTM_Unit(0, encpar_rm, model_options['batch_size'], model_options['dim_proj'],
                         model_options['dim_proj'], model_options['output_dim'], model_options['l2_decay']))

    model = Model()
    model.add(net_lu)
    model.add(net_ru)
    model.add(net_rv)
    model.add(net_rm)

    # add hierarchical encode layers
    model.add(net_en_hiera)

    # add hierarchical decode layers
    model.add(net_ru_high)
    model.add(net_rv_high)
    model.add(net_rm_high)

    model.connect(net_en_hiera, net_ru_high)
    model.connect(net_en_hiera, net_rv_high)
    model.connect(net_en_hiera, net_rm_high)

    return model

def run_blstm(
    dim_proj=256,  # LSTM number of hidden units.
    dim_frame=4096, # feature dimension of image frame in the video
    output_dim = 4096,
    v_length = 24, # video length or number of frames
    max_iter=100,  # The maximum number of epoch to run
    l2_decay=0.0001,  # Weight decay for model params.
    lrate=0.0001,  # Learning rate for SGD, Adam
    lamb = 0.2,
    optimizer='SGD',  # SGD, Adam available
    saveto='pairwise-blstm_model.npz',  # The best model will be saved there
    dispFreq=2,  # Display to stdout the training progress every N updates
    validFreq=20,  # Compute the validation error after this number of update.
    saveFreq=2,  # Save the parameters after every saveFreq updates
    batch_size=256,  # The batch size during training.
    valid_batch_size=20,  # The batch size used for validation/test set.
    test_batch_size = 1024,
    weights=[1./3.,1./3.,1./3.],  # The Weights for forwoad and backward reconstruction and mean value reconstruction
    pairwise_weight = 0.999,
    reload_model=False,  # If reload model from saveto.
    is_train=False,
    test_step = 1,
    hiera_step = 2,
    train_data_file_path = '/mnt/data2/lixiangpeng/dataset/features/FCV/fcv/',
    test_data_file_path = '/mnt/data2/lixiangpeng/dataset/features/FCV/fcv/',
    #train_data_file_path = './',
    #test_data_file_path = './',
    train_splits_num = 1,
    test_splits_num = 1,
    record_path = './records.txt',
    SS_path = '/mnt/data2/lixiangpeng/dataset/features/FCV/SimilarityInfo/Sim_K1_10_K2_5_fcv.pkl'

):
    model_options = locals().copy()
    if reload_model:
        print "Reloading model options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)
    print "model options", model_options

    test_data = DataHelper.DataHelper(model_options['v_length'], model_options['valid_batch_size'],model_options['dim_frame'],
            data_file= './data/fcv_test_demo.h5', train=False)


    model = build_model(model_options)

    if reload_model:
        model.reload_params(saveto)

    model.compile(model_options)


    if is_train:
        model.train(model_options['train_data_file_path'], test_data, model_options)
    else:
        model.test(test_data, model_options)
    
if __name__ == '__main__':
    # is_train:
    #   True for training model,
    #   False for testing model and generate hidden vectors for test data.
    run_blstm(is_train=True)

