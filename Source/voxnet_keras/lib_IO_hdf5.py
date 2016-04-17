from __future__ import print_function
import numpy as np
import scipy.io
import os
import h5py
from sklearn import utils

utils.shuffle()

def save_dataset_as_hdf5(dirname_data, fname_save, class_name_to_id):

    classnames = set(class_name_to_id.keys())

    #find out how many files for test and train exist. Speed Reasons
    train_size = 0
    test_size = 0
    for dirpaths, dirs, fnames  in os.walk(dirname_data):
        for fname in fnames:
            if fname.endswith('.mat'):
                pos1 = fname.find('_')
                #the correct files have a _ before the rotation
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
    info_train = np.zeros([train_size,3],dtype=np.uint32)
    feat_test = np.zeros([test_size,1,32,32,32],dtype=np.uint8)
    lab_test = np.zeros([test_size,],dtype=np.uint32)
    info_test = np.zeros([test_size,3],dtype=np.uint32)

    train_it = 0
    test_it = 0
    classnames = set(class_name_to_id.keys())

    for dirpaths, dirs, fnames  in os.walk(dirname_data):
        for fname in fnames:
            if fname.endswith('.mat'):
                pos1 = fname.find('_')

                if pos1 is -1:
                    continue

                #check if loaded class is one of the required classes
                classname = fname[:pos1]
                if classname not in classnames:
                    continue

                #find set_type info
                pos2 = dirpaths.rfind('/')
                set_type = dirpaths[pos2+1:]

                #find obj_ID info
                pos3 = fname.find('_',pos1+1)
                obj_id = int(fname[pos1+1:pos3])

                #find rotation_Id info
                pos4 = fname.find('.', pos3 + 1)
                rot_id = int(fname[pos3+1:pos4])

                #encode class and add to labels
                label = int(class_name_to_id[classname])

                #load mat, reshape to 32x32x32 array and add to save to .npy file
                arr = scipy.io.loadmat(os.path.join(dirpaths,fname))['instance'].astype(np.uint8)

                if set_type == "train":
                    feat_train[train_it,0,1:-1,1:-1,1:-1] = arr
                    lab_train[train_it] = label
                    info_train[train_it,:] = [label, obj_id, rot_id]
                    train_it += 1
                elif set_type == "test":
                    feat_test[test_it,0,1:-1,1:-1,1:-1] = arr
                    lab_test[test_it] = label
                    info_test[test_it,:] = [label, obj_id, rot_id]
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
    train.create_dataset("info_train", [train_size,3],
                                        dtype = np.uint32,
                                        chunks = True,
                                        compression="gzip",
                                        data=info_train)
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
    test.create_dataset("info_test", [test_size,3],
                                      dtype = np.uint32,
                                      chunks = True,
                                      compression="gzip",
                                      data=info_test)
    openfile.close()

#loader needs a HDF file with a subgroup of name set_type
#  which holds a "labels_"+set_type and "features_"+set_type dataset
class Loader_hdf5:
    def __init__(self, fname, set_type = "train",
                 batch_size = 12*128, num_batches = None,
                 shuffle = False, valid_split = None):
        openfile = h5py.File(fname)

        lab = openfile[set_type + "/labels_" + set_type]
        self._labels = np.zeros(lab.shape,dtype=np.uint32)
        lab.read_direct(self._labels)

        feat = openfile[set_type + "/features_" + set_type]
        self._features = np.zeros(feat.shape,dtype=np.uint8)
        feat.read_direct(self._features)

        try:
            info = openfile[set_type + "/info_" + set_type]
            self._info = np.zeros(info.shape,dtype=np.uint32)
            info.read_direct(self._info)
            self._has_rot = True
        except IOError:
            self._has_rot = False

        openfile.close()

        self._batch_size = batch_size
        self._pos = 0
        self._set_type = set_type

        if num_batches is not None and num_batches*batch_size < self._labels.shape[0]:
            self._max_pos = num_batches*batch_size
        else:
            self._max_pos = self._labels.shape[0]

        self.sort_by_rotations()

        if shuffle is True:
            self.shuffle_data()

        if valid_split is not None:
            self._valid_size = valid_split
            self.validation_split()
        else:
            self._features_train = self._features
            self._labels_train = self._labels

        self._features = None
        self._labels = None


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._pos >= self._max_pos:
            raise StopIteration
        features = self._features_train[self._pos:self._pos+self._batch_size,:,:,:,:]
        labels = self._labels_train[self._pos:self._pos+self._batch_size]

        self._pos += self._batch_size
        return features, labels

    def change_set_type(self,set_type):
        self._set_type = set_type

    def change_batch_size(self,batch_size):
        self._batch_size = batch_size

    def change_validation_size(self,valid_split):
        self._valid_size = valid_split
        self.validation_split()

    def shuffle_data(self):
        if self._has_rot is True:
            step_size = np.amax(self._info[:,2]) - np.amin(self._info[:,2]) + 1
        else:
            step_size = 1
        # Fisher-Yatest shuffle assuming that rotations of one obj are together
        for fy_i in range(self._labels.shape[0]-1,1 + step_size,-1 * step_size):
            fy_j = np.random.randint(1,int((fy_i+1)/step_size) + 1) * step_size - 1
            if fy_j - step_size < 0:
                self._features[fy_i:fy_i-step_size:-1,:,:,:,:],self._features[fy_j::-1,:,:,:,:] =\
                    self._features[fy_j::-1,:,:,:,:], self._features[fy_i:fy_i-step_size:-1,:,:,:,:].copy()
                self._labels[fy_i:fy_i-step_size:-1],self._labels[fy_j::-1] =\
                    self._labels[fy_j::-1], self._labels[fy_i:fy_i-step_size:-1].copy()
                self._info[fy_i:fy_i-step_size:-1,:],self._info[fy_j::-1,:] =\
                    self._info[fy_j::-1,:], self._info[fy_i:fy_i-step_size:-1,:].copy()
            else:
                self._features[fy_i:fy_i-step_size:-1], self._features[fy_j:fy_j-step_size:-1]=\
                    self._features[fy_j:fy_j-step_size:-1], self._features[fy_i:fy_i-step_size:-1].copy()
                self._labels[fy_i:fy_i-step_size:-1], self._labels[fy_j:fy_j-step_size:-1]=\
                    self._labels[fy_j:fy_j-step_size:-1], self._labels[fy_i:fy_i-step_size:-1].copy()
                self._info[fy_i:fy_i-step_size:-1], self._info[fy_j:fy_j-step_size:-1]=\
                    self._info[fy_j:fy_j-step_size:-1], self._info[fy_i:fy_i-step_size:-1].copy()

    def validation_split(self):
        if self._has_rot is True:
            step_size = np.amax(self._info[:,2]) - np.amin(self._info[:,2]) + 1
        else:
            step_size = 1
        split_pos = int(int((self._labels.shape[0]/step_size)*(1-self._valid_size))*step_size)
        self._features_train = self._features[:split_pos,:,:,:,:]
        self._labels_train = self._labels[:split_pos]
        self._features_valid = self._features[split_pos:,:,:,:,:]
        self._labels_valid = self._labels[split_pos:]

    def sort_by_rotations(self):
        sort_scheme = np.argsort(self._info[:,1],axis = 0)
        self._info = self._info[sort_scheme,:]
        self._features = self._features[sort_scheme,:,:,:,:]
        self._labels = self._labels[sort_scheme]

    def return_valid_set(self):
        return self._features_valid, self._labels_valid

    def return_pos(self):
        return self._pos

