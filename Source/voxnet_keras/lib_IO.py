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


###################################npytar
PREFIX = 'data/'
SUFFIX = '.npy.z'

class NpyTarWriter(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'w|')

    def add(self, arr, name):

        sio = StringIO.StringIO()
        np.save(sio, arr)
        zbuf = zlib.compress(sio.getvalue())
        sio.close()

        zsio = StringIO.StringIO(zbuf)
        tinfo = tarfile.TarInfo('{}{}{}'.format(PREFIX, name, SUFFIX))
        tinfo.size = len(zbuf)
        tinfo.mtime = time.time()
        zsio.seek(0)
        self.tfile.addfile(tinfo, zsio)
        zsio.close()

    def close(self):
        self.tfile.close()


class NpyTarReader(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'r|')

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        entry = self.tfile.next()
        if entry is None:
            raise StopIteration()
        name = entry.name[len(PREFIX):-len(SUFFIX)]

        fileobj = self.tfile.extractfile(entry)

        buf = zlib.decompress(fileobj.read())

        arr = np.load(BytesIO(buf))
        return arr, name

    def close(self):
        self.tfile.close()

    def elements(self):
        n_elem = 0
        for it, (x, name) in enumerate(self):
            n_elem += 1
        return n_elem


####################################################
#load CAD data Mat files and save as npy Files
#take labelnames from class_names, class names_to_id has to be a directory['name':'1']
def save_dataset_as_npy(fname_data, fname_save, class_name_to_id):
    #iterate through all .mat files with CAD data
    features = {'train': 0,
                'test': 0}
    labels = {'train': 0,
              'test': 0}
    classnames = set(class_name_to_id.keys())
    for dirpaths, dirs, fnames  in os.walk(fname_data):
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

                if labels[set_type] is 0:
                    labels[set_type] = np.zeros([1,], dtype=np.uint32)
                    labels[set_type][0] = Id
                else:
                    labels[set_type] = np.concatenate((labels[set_type],Id), axis = 0)

                #load mat, reshape to 32x32x32 array and add to features
                arr = scipy.io.loadmat(os.path.join(dirpaths,fname))['instance'].astype(np.uint8)
                arrpad = np.zeros([1,1,32,32,32], dtype=np.uint8)
                arrpad[0,0,1:-1,1:-1,1:-1] = arr
                if features[set_type] is 0:
                    features[set_type] = np.zeros([1,1,32,32,32], dtype=np.uint8)
                    features[set_type][0,:,:,:,:] = arrpad
                else:
                    features[set_type] = np.concatenate((features[set_type],arrpad), axis = 0)

    if features['train'].shape[0] is not labels['train'].shape[0]:
        print("train went wrong {0} : {1}".format(features['train'].shape[0],labels['train'].shape[0]))
    if features['test'].shape[0] is not labels['test'].shape[0]:
        print("test went wrong{0} : {1}".format(features['test'].shape[0],labels['test'].shape[0]))

    print("found {0} train datasets and {1} test datasets".format(labels['train'].shape[0],labels['test'].shape[0]))

    outfile = open(fname_save,'wb')
    np.savez(outfile,
             features_train = features['train'],
             labels_train = labels['train'],
             features_test = features['test'],
             labels_test = labels['test']
             )
    outfile.close()


#####################################################
#load data from npz file and return features_train,labels_train,features_test,labels_test
def load_data(fname_data):
    f_open = open(fname_data, 'r')

    data_npz = np.load(f_open)
    #return features & labels
    return (data_npz['features_train'], data_npz['labels_train'],
            data_npz['features_test'], data_npz['labels_test'])

# use only if cross-validation not possible
def train_valid_split(dataset, labels, valid_size):
    data_train, data_valid, label_train, label_valid = train_test_split(dataset, labels, test_size= valid_size)
    return data_train, data_valid, label_train, label_valid

#shuffle dataset & labels
def shuffle(dataset, labels):
    dataset_sfl, labels_sfl = shuffle(dataset, labels)
    return dataset_sfl, labels_sfl

