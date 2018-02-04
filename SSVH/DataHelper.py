import cPickle
import gzip
import os
import sys

import numpy as np
import h5py as h5

class DataHelper(object):
    def __init__(self,seq_length,batch_size,image_size,data_file,train):
      self.seq_length_ = seq_length
      self.batch_size_ = batch_size
      self.frame_size_ = image_size
      self.data_file_ = data_file
      self.istrain_ = train
      self.row_ = 0

      try:
        r = h5.File(self.data_file_, 'r')
        self.data_ = r['feats']
        '''
        X = self.data_
        m = np.mean(X, axis=(0,1))
        brodcast_m = np.reshape(m, (1,1,4096))
        std = np.mean(np.square(X - brodcast_m) + 1e-10, axis=(0,1))
        std = np.sqrt(std)
        brodcast_std = np.reshape(std, (1,1,4096))
        X_normed = (X - brodcast_m) / (brodcast_std)
        self.data_ = X_normed
        '''
        self.data_size_ = self.data_.shape[0]
        print 'data size',self.data_.shape
        self.idx_ = range(self.data_size_)

        if self.istrain_:
          np.random.shuffle(self.idx_)

      except:
        print 'Please set the correct path to the dataset'
        sys.exit()


    def Reset(self):
      self.row_ = 0
      if self.istrain_:
        np.random.shuffle(self.idx_)
      pass

    def GetBatch(self):
      endi = min([self.row_+self.batch_size_, self.data_size_])
      idx = self.idx_[self.row_:endi]
      idx = np.sort(idx)
      minibatch = self.data_[idx,:,:]
      if minibatch.shape[0] < self.batch_size_:
        minibatch = np.row_stack((minibatch,np.float32(np.zeros((self.batch_size_-minibatch.shape[0],self.seq_length_,self.frame_size_)))))
      self.row_ = endi

      if self.row_ == self.data_size_:
        self.Reset()

      return minibatch
