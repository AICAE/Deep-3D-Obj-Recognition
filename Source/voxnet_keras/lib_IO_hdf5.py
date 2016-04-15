from __future__ import print_function
import numpy as np
import scipy.io
import os
import h5py


def save_dataset_as_hdf5(dirname_data, fname_save, class_name_to_id):

    classnames = set(class_name_to_id.keys())

    #find out how many files for test and train exist. Speed Reasons
    train_size = 0
    test_size = 0
    for dirpaths, dirs, fnames  in os.walk(dirname_data):
        for fname in fnames:
            if fname.endswith('.mat'):
                pos1 = fname.find('_')
                #the correct files have a _ after the class name
                if pos1 is -1:
                    continue

                #check if loaded class is one of the required classes
                classname = fname[:pos1]
                if classname not in classnames:
                    continue

                pos2 = dirpaths.rfind('/')
                set_type = dirpaths[pos2+1:]
                if set_type == 'train':
                    train_size += 1
                elif set_type == 'test':
                    test_size += 1

    print("found {0} train files and {1} test files".format(train_size,test_size))

    feat_train = np.zeros([train_size,1,32,32,32],dtype=np.uint8)
    lab_train = np.zeros([train_size,],dtype=np.uint32)
    feat_test = np.zeros([test_size,1,32,32,32],dtype=np.uint8)
    lab_test = np.zeros([test_size,],dtype=np.uint32)


    train_it = 0
    test_it = 0
    classnames = set(class_name_to_id.keys())
    for dirpaths, dirs, fnames  in os.walk(dirname_data):
        for fname in fnames:
            if fname.endswith('.mat'):
                pos1 = fname.find('_')
                #the correct files have a _ after the class name
                if pos1 is -1:
                    continue

                #check if loaded class is one of the required classes
                classname = fname[:pos1]
                if classname not in classnames:
                    continue

                pos2 = dirpaths.rfind('/')
                set_type = dirpaths[pos2+1:]

                #encode class and add to labels
                Id = np.zeros([1,], dtype=np.uint32)
                Id[-1] = int(class_name_to_id[classname])

                if set_type == "train":
                    lab_train[train_it] = Id
                elif set_type == "test":
                    lab_test[test_it] = Id

                #load mat, reshape to 32x32x32 array and add to save to .npy file
                arr = scipy.io.loadmat(os.path.join(dirpaths,fname))['instance'].astype(np.uint8)

                if set_type == "train":
                    feat_train[train_it,0,1:-1,1:-1,1:-1] = arr
                    train_it += 1
                elif set_type == "test":
                    feat_test[test_it,0,1:-1,1:-1,1:-1] = arr
                    test_it += 1


    print("found {0} train datasets and {1} test datasets".format(lab_train.shape,lab_test.shape))

    #create hdf5 dataset storage and iterate through all files and save them. FileSize Reason
    openfile = h5py.File(fname_save, "w")
    train = openfile.create_group("train")
    train.create_dataset("features_train", [train_size,1,32,32,32],
                                          dtype = np.uint8,
                                          chunks = True,
                                          compression="gzip",
                                          data=feat_train)
    train.create_dataset("labels_train", [train_size,],
                                        dtype = np.uint32,
                                        chunks = True,
                                        compression="gzip",
                                        data=lab_train)
    test = openfile.create_group("test")
    test.create_dataset("features_test", [test_size,1,32,32,32],
                                        dtype = np.uint8,
                                        chunks = True,
                                        compression="gzip",
                                        data=feat_test)
    test.create_dataset("labels_test", [test_size,],
                                      dtype = np.uint32,
                                      chunks = True,
                                      compression="gzip",
                                      data=lab_test)
    openfile.close()

class loader_hdf5:
    def __init__(self, fname, set_type = "train", batch_size = 12*128):
        openfile = h5py.File(fname)

        lab = openfile[set_type + "/labels_" + set_type]
        self._labels = np.zeros(lab.shape,dtype=np.uint8)
        lab.read_direct(self._labels)

        feat = openfile[set_type + "/features_" + set_type]
        self._features = np.zeros(feat.shape,dtype=np.uint8)
        feat.read_direct(self._features)

        self._batch_size = batch_size
        self._pos = 0
        self._set_type = set_type
        self._max_pos = self._labels.shape[0] - self._batch_size

        openfile.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._pos >= self._max_pos:
            raise StopIteration
        features = self._features[self._pos:self._pos+self._batch_size,:,:,:,:]
        labels = self._labels[self._pos:self._pos+self._batch_size]

        self._pos += self._batch_size
        return features, labels

    def change_set_type(self,set_type):
        self._set_type = set_type

    def change_batch_size(self,batch_size):
        self._batch_size = batch_size

    def return_pos(self):
        return self._pos