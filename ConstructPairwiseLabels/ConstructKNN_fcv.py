# -*- coding: utf-8 -*
import h5py
import numpy as np
import cPickle as pkl
import time
import scipy.io as sio

file = h5py.File('/data2/lixiangpeng/dataset/fcv/fcv_train_feats.h5')

print file.keys()
feats = file['feats']
print "data transfering..."
time1 = time.time()
feats_array = np.array(feats)
time2 = time.time()
print "data transfering costs：",time2-time1

print "mean pooling computing..."
feats_meanpooling = feats_array.mean(1)
time3 = time.time()
print "mean pooling costs: ",time3-time2
print "mean pooling shape: ",feats_meanpooling.shape

# f = open('yfcc_train_feats_1.pkl','w')
# pkl.dump(feats_meanpooling,f)
# f.close()

# file = open('yfcc_train_feats_1.pkl')
# feats_meanpooling = pkl.load(file)

# we use cosine distance to measure the similarity of two samples

fcv_KNN = []
topN = 1000
[sample_num,sample_dim] = feats_meanpooling.shape
S = -1.*np.ones((sample_num,sample_num))
for i in range(sample_num):
    feat_a = feats_meanpooling[i,:]
    ss = []
    n = []
    #ac =
    aui = np.sqrt(np.sum(feat_a**2))
    for j in range(sample_num):
        feat_b = feats_meanpooling[j,:]
        sim = np.sum(feat_a*feat_b)/(aui*np.sqrt(np.sum(feat_b**2)))
        ss.append(sim)
#        n.append(Label[j])
    ss = np.array(ss)
    index = np.argsort(ss*-1.)
    fcv_KNN.append(index[:topN+1])
    print i
np.savez('fcv_KNN.npz',topK=fcv_KNN)

