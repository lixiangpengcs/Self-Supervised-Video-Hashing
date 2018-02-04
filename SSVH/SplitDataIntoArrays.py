import random
import numpy
import h5py


def indexContent(TrainData,index):
    content = []
    for i in index:
        a= TrainData.data_[i]
        content.append(a)
    content = numpy.array(content)[:,:TrainData.seq_length_,:]
    
    #print content.shape
    if content.shape[0]<TrainData.batch_size_:
        content = numpy.row_stack((content, numpy.float32(
            numpy.zeros((TrainData.batch_size_ - content.shape[0], TrainData.seq_length_, TrainData.frame_size_)))))
    # batch_size *
    return content

def SplitDataIntoArrays(Data,step):
    X = []
    array = range(len(Data))
    random.shuffle(array)
    length = len(array)
    m = length/step
    if length%step == 0:
        m = m
    else:
        m += 1
    for i in range(0,m):
        print i
        if i==(m-1):
            X.append(indexContent(Data,array[i*step:]))
        else:
            X.append(indexContent(Data,array[i*step:(i+1)*step]))
    return numpy.array(X)

if __name__ == '__main__':
    file = h5py.File('yfcc_train_feats_1.h5', 'r')
    feats = file['feats']
    print 'Load Done!'
    X = SplitDataIntoArrays(feats,256)
    for i in X:
        print numpy.array(i).shape
