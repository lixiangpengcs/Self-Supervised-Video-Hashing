import numpy as np
import scipy.io as sio
import os
import cPickle as pkl

L = sio.loadmat('/mnt/data2/lixiangpeng/dataset/features/FCV/fcv/fcv_train_labels.mat')['labels']
data = np.load('fcv_KNN.npz')['topK']
outpath = '/mnt/data2/lixiangpeng/dataset/features/FCV/SimilarityInfo/'
#
# L = sio.loadmat('./valid_framework.mat')['train_labels']
# print 'L.shape: ',L.shape
# data = np.load('fcv_KNN.npz')['topK']

print "loading data done!"

#test
K1 = 10
K2 = 5
size = len(L)
LL = []
ans = 0
SimilarityMatrix =  np.zeros((size,size),dtype=np.bool)
for i in xrange(size):
    si = set(data[i][:K1+1])
    gth = L[i]
    nums = []
    idx = []
    for j in xrange(size):
        sj = set(data[j][:K1+1])
        intersection = si & sj
        nums.append(len(intersection))
        idx.append(j)

    nums = np.array(nums)
    indx = np.argsort(nums*-1)[:K2]
    sp = set()

    for ii in indx:
        idx_ = idx[ii]
        sp = sp|set(data[idx_][:K1+1])

    cnt = 0.
    for ii in sp:
        #if gth == L[ii]:
        #print np.dot(gth,L[ii])
        if np.dot(gth,L[ii])!=0:
            cnt += 1
    ans +=(cnt/len(sp))
    #print ans/(i+1.),cnt,len(sp),i
    for j in sp:
        SimilarityMatrix[i][j] = True
        SimilarityMatrix[j][i] = True
    if i%10 == 0:
        print ans / (i + 1.), len(sp), cnt,i in sp

#sio.savemat("Sim_K1_20_K2_30.mat",{'sim':SimilarityMatrix})
file = open(outpath+'Sim_K1_'+str(K1)+'_K2_'+str(K2)+'_fcv.pkl','w')
pkl.dump(SimilarityMatrix,file)
file.close()




