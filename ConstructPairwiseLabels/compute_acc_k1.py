import numpy as np
import scipy.io as sio
topN = 20

array = np.load('fcv_KNN.npz')['topK']   #train_size * 1001
[size_r,size_c] = array.shape
SimilarityMatrix =  np.zeros((size_r,size_r))

L = sio.loadmat('/data2/lixiangpeng/dataset/fcv/fcv_train_labels.mat')['labels'] # train_size * 239
ans = 0.0
for i in range(size_r):
    cnt = 0.
    for j in range(topN):

        pre = array[i][j]
        print np.dot(L[i],L[pre])
        if np.dot(L[i],L[pre])!=0:
            cnt += 1
    ans += cnt/topN
    print i,ans/(i+1),cnt

