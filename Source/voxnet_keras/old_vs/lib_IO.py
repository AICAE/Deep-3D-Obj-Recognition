from __future__ import print_function
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from io import BytesIO
import tarfile
import time
import zlib
import scipy.io
from path import Path
import importlib
import os
import h5py


####################################################
#load CAD data Mat files and save as npy Files
#take labelnames from class_names, class names_to_id has to be a directory['name':'1']
def save_dataset_as_npy(dirname_data, dirname_save, class_name_to_id):
    #iterate through all .mat files with CAD data
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
                
                Id = int(class_name_to_id[classname])
                fname_save = dirpaths + set_type + "/" +  "{:3d}_".format(Id) + fname 

                #load mat, reshape to 32x32x32 array and add to save to .npy file
                arr = scipy.io.loadmat(os.path.join(dirpaths,fname))['instance'].astype(np.uint8)
                arrpad = np.zeros([1,1,32,32,32], dtype=np.uint8)
                arrpad[0,0,1:-1,1:-1,1:-1] = arr
                features = np.zeros([1,1,32,32,32], dtype=np.uint8)
                features [0,0,:,:,:] = arrpad
                outfile = open(fname_save,'wb')
                np.save(outfile, features[set_type])
                outfile.close()

def save_dataset_as_npz(dirname_data, fname_save, class_name_to_id):
    #iterate through all .mat files with CAD data
    features = {'train': np.zeros([100000,1,32,32,32]),
                'test': np.zeros([40000,1,32,32,32])}
    labels = {'train': np.zeros([100000,]),
              'test': np.zeros([40000,])}
    it = 0;
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
                labels[set_type][it] = Id

                #load mat, reshape to 32x32x32 array and add to save to .npy file
                arr = scipy.io.loadmat(os.path.join(dirpaths,fname))['instance'].astype(np.uint8)
                arrpad = np.zeros([1,1,32,32,32], dtype=np.uint8)
                arrpad[0,0,1:-1,1:-1,1:-1] = arr
                features[set_type][it,:,:,:,:] = arrpad

                it += 1

    print("found {0} train datasets and {1} test datasets".format(labels['train'].shape[0],labels['test'].shape[0]))

    outfile = open(fname_save,'wb')
    #Maybe change it to creating to a single .rar folder for easier transportability
    np.savez(outfile,
             features_train = features['train'],
             labels_train = labels['train'],
             features_test = features['test'],
             labels_test = labels['test']
             )
    outfile.close()

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
                if set_type is "train":
                    train_size += 1
                elif set_type is "test":
                    test_size += 1

    #create hdf5 dataset storage and iterate through all files and save them.
    openfile = h5py.File(fname_save, "w")
    train = openfile.create_group("train")
    features_train = train.create_dataset("features_train", [train_size,1,32,32,32],
                                          dtype = np.unit8,
                                          chunks = True,
                                          compression="gzip")
    labels_train = train.create_dataset("labels_train", [train_size,],
                                        dtype = np.unit32,
                                        chunks = True,
                                        compression="gzip")
    test = openfile.create_group("labels")
    features_test = test.create_dataset("features_train", [test_size,1,32,32,32],
                                        dtype = np.unit8,
                                        chunks = True,
                                        compression="gzip")
    labels_test = test.create_dataset("labels_train", [test_size,],
                                      dtype = np.unit32,
                                      chunks = True,
                                      compression="gzip")

    it = 0;
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
                if set_type is "train":
                    labels_train[it] = Id
                elif set_type is "test":
                    labels_test[it] = Id

                #load mat, reshape to 32x32x32 array and add to save to .npy file
                arr = scipy.io.loadmat(os.path.join(dirpaths,fname))['instance'].astype(np.uint8)
                if set_type is "train":
                    features_train[it,0,1:-1,1:-1,1:-1] = arr
                elif set_type is "test":
                    features_test[it,0,1:-1,1:-1,1:-1] = arr

                it += 1

    print("found {0} train datasets and {1} test datasets".format(labels_train[0],labels_test.shape[0]))

    openfile.close()

#####################################################
#load data from npz file and return features_train,labels_train,features_test,labels_test
def load_data_npz(fname_data):
    f_open = open(fname_data, 'r')
    data_npz = np.load(f_open)
    f_open.close()

    #return features & labels
    return (data_npz['features_train'], data_npz['labels_train'],
            data_npz['features_test'], data_npz['labels_test'])


class Loader_npy(object):
    def __init__(self, dirname, class_name_to_id, batch_size = 12 * 128 , num_batch = None, set_type = None, shffle = False):
        self._walker = os.walk(dirname,topdown = True, onerror = None, followlinks = False)
        self._features = np.zeros([batch_size, 1, 32, 32, 32], dtype = np.uint8)
        self._labels = np.zeros([batch_size,], dtype= np.uint32)
        self._iter = 0
        self._batch_size = batch_size
        self._fnames = os.listdir(dirname)
        if shffle is True:
            self._fnames = os.listdir(dirname)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        for it in range(0,self._batch_size):
            self._iter += 1


            openfile = open(dirpath + fname, 'rb')
            self._features[it,:,:,:,:] = np.load(openfile, dtype=np.uint8)
            pos1 = fname.find('_')
            self._labels[it] = int(fname[:pos1])
            
        
        return self._features, self._labels

def load_data_npy(dirname_data, batch_size, class_name_to_id, set_type = None):
    
    classnames = set(class_name_to_id.keys())
    if set_type is not None:
        dirname_data = dirname_dat + "/" + set_type
    for dirpaths, dirs, fnames  in os.walk(dirname_data):
        for fname in fnames:
            pos1 = fname.find('_')
                #the correct files have a _ after the class name
                if pos1 is -1:
                    continue
                
                #check if loaded class is one of the required classes
                classname = fname[:pos1]
                if classname not in classnames:
                    continue
                
                np.load(fname)
                
# use only if cross-validation not possible
def train_valid_split(dataset, labels, valid_size):
    data_train, data_valid, label_train, label_valid = train_test_split(dataset, labels, test_size= valid_size)
    return data_train, data_valid, label_train, label_valid

#shuffle dataset & labels
def shuffle(dataset, labels):
    dataset_sfl, labels_sfl = shuffle(dataset, labels)
    return dataset_sfl, labels_sfl

