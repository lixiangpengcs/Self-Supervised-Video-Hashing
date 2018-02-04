import numpy as np
import DataHelper
import io
import time
import scipy.io as sio
import tools

class Array():
    def __init__(self):
        pass
    def setmatrcs(self,matrics):
        self.matrics = matrics

    def concate_v(self,matrics):
        self.matrics = np.vstack((self.matrics,matrics))

    def getmatrics(self):
        return self.matrics

def sign(data):
    BinaryCode = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]>0:
                BinaryCode[i,j] = 1
            else:
                BinaryCode[i,j] = -1

    return BinaryCode

def cat_ap_topK(cateTrainTest, HammingRank, M_set):
    precision = np.zeros((len(M_set), 1))
    recall = np.zeros((len(M_set), 1))

    numTest = cateTrainTest.shape[1]

    for i, K in enumerate(M_set):
        precisions = np.zeros((numTest, 1))
        recalls = np.zeros((numTest, 1))

        topk = HammingRank[:K, :]

        for qid in range(numTest):
            retrieved = topk[:, qid]
            rel = cateTrainTest[retrieved - 1, qid]
            retrieved_relevant_num = np.sum(rel)
            real_relevant_num = np.sum(cateTrainTest[:, qid])

            precisions[qid] = retrieved_relevant_num/(K*1.0)
            recalls[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

        precision[i] = np.mean(precisions)
        recall[i] = np.mean(recalls)

    return precision, recall

def cat_apcal(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    if num_return_NN: num_return_NN = numTrain

    apall = np.zeros((numTest, 1))

    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(numTrain):
            if cateTrainTest[query[rid] - 1, qid]:
                x += 1
                p += x/(rid*1.0+1.0)

        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(x*1.0)

    return np.mean(apall)


def test(_encoder,options,uidx):
    print 'loading test data...'
    hashcode_array = Array()
    h_array = Array()
    label_array = Array()
    lines_num = 0
    for i in range(1,options['test_splits_num']+1):
        file_name = 'fcv_test_feats.h5'
        labels_name = 'fcv_test_labels.mat'
        print 'loading ',file_name
        test_data = DataHelper.DataHelper(options['v_length'], options['batch_size'],
                                          options['dim_frame'],
                                          data_file= options['test_data_file_path']+file_name, train=True)
        labels = sio.loadmat(options['test_data_file_path']+labels_name)['labels']
        lines_num += test_data.data_size_
        if i ==1:
            label_array.setmatrcs(labels)
        else:
            label_array.concate_v(labels)
        print 'data_size: ',test_data.data_size_
        print 'batch_size: ',test_data.batch_size_

        batch_num = test_data.data_size_/options['test_batch_size']
        if test_data.data_size_ % options['test_batch_size'] == 0:
            batch_num = batch_num
        else:
            batch_num += 1

        for batch_idx in range(batch_num):
            print 'batch_idx: ',batch_idx
            time1 = time.time()
            if batch_idx == (batch_num-1):
                X = test_data.data_[batch_idx*options['test_batch_size']:][:,:options['v_length'],:]
                X = np.row_stack((X,np.float32(np.zeros((options['test_batch_size']-X.shape[0],options['v_length'],options['dim_frame'])))))
            else:
                X = test_data.data_[batch_idx*options['test_batch_size']:(batch_idx+1)*options['test_batch_size']][:,:options['v_length'],:]
            time2 = time.time()
            print 'fetching data costs: ',time2-time1
            print 'batch data shape: ',X.shape
            my_H = _encoder(X,np.zeros((X.shape[0], options['dim_proj']), dtype=np.float32),
                                np.zeros((X.shape[0], options['dim_proj']), dtype=np.float32),
                                np.zeros((X.shape[0], options['dim_proj']), dtype=np.float32),
                                np.zeros((X.shape[0], options['dim_proj']), dtype=np.float32))
            time3 = time.time()
            print 'forward costs: ',time3-time2
            print 'my_H: ',my_H.shape
            BinaryCode = sign(my_H)
            if i == 1 and batch_idx == 0:
                hashcode_array.setmatrcs(BinaryCode)
                h_array.setmatrcs(my_H)
            else:
                hashcode_array.concate_v(BinaryCode)
                h_array.concate_v(my_H)

        hashcode_array.setmatrcs(hashcode_array.getmatrics()[:lines_num])
        h_array.setmatrcs(hashcode_array.getmatrics()[:lines_num])
        print 'hashcode shape:',hashcode_array.getmatrics().shape
    #sio.savemat(str(options['dim_proj'])+'_'+'hashcode_' + str(uidx) + '.mat', {'hashcode': hashcode_array.getmatrics()})
    #sio.savemat(str(options['dim_proj'])+'_'+'h_' + str(uidx) + '.mat', {'h': h_array.getmatrics()})

    test_hashcode = hashcode_array.getmatrics()
    print 'test_hashcode: ',test_hashcode.shape

    test_hashcode = np.matrix(test_hashcode)
    time1 = time.time()
    Hamming_distance = 0.5*(-np.dot(test_hashcode,test_hashcode.transpose())+options['dim_proj'])
    time2 = time.time()
    print 'hamming distance computation costs: ',time2-time1
    HammingRank = np.argsort(Hamming_distance, axis=0)
    time3 = time.time()
    print 'hamming ranking costs: ',time3-time2

    labels = label_array.getmatrics()
    print 'labels shape: ',labels.shape
    sim_labels = np.dot(labels, labels.transpose())
    time6 = time.time()
    print 'similarity labels generation costs: ', time6 - time3

    records = open('map.txt','w+')
    maps = []
    for i in range(5,105,5):
        map = tools.mAP(sim_labels, HammingRank,i)
        maps.append(map)
        records.write('epoch: '+str(uidx)+'\ttopK: '+str(i)+'\tmap: '+str(map)+'\n')
        print 'i: ',i,' map: ', map,'\n'
    time7 = time.time()
    records.close()
    print 'computing processing costs: ', time7 - time6

    return maps
if __name__ == '__main__':
    #for test
    arr = Array()
    a = np.zeros((3,2))
    arr.setmatrcs(a)
    for i in range(7):
        if i%3 == 0:
            a = np.ones([3,2])
            arr.concate_v(a)
        else:
            a = np.zeros([3,2])
            arr.concate_v(a)
    b = arr.getmatrics()
    print b.shape
    print b[:,1]

