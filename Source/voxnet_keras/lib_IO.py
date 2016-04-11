from __future__ import print_function
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from io import StringIO
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
        arr = np.load(StringIO.StringIO(buf))
        return arr, name

    def close(self):
        self.tfile.close()

    def elements(self):
        n_elem = 0
        for it, (x, name) in enumerate(self):
            n_elem += 1
        return n_elem


####################################################
#load CAD data Mat files and save as Tarfile
#take labelnames from class_names, class names has to be a directory['name':'1']
def save_Dataset_as_tar(fname_data, fname_tar, class_name_to_id):
    #import configfile
    importlib.import_module('fname_cfg')

    #open directory with data
    base_dir = Path(fname_data).expand()

    #create seperate train & test sets
    records = {'train': [], 'test': []}

    #iterate through all .mat files with CAD data
    for fname in os.listdir('/home/tg/Downloads/volumetric_data'):
        if fname.endswith('.mat'):
            elts = fname.splitall()
            instance_rot = Path(elts[-1]).stripext()
            instance = instance_rot[:instance_rot.rfind('_')]
            rot = int(instance_rot[instance_rot.rfind('_')+1:])
            split = elts[-2]
            classname = elts[-4].strip()
            if classname not in class_name_to_id:
                continue
    records[split].append((classname, instance, rot, fname))

    writer = NpyTarWriter(fname)
    for (classname, instance, rot, fname) in records:
        class_id = int(class_name_to_id[classname])
        name = '{:03d}.{}.{:03d}'.format(class_id, instance, rot)
        arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
        arrpad = np.zeros((32,)*3, dtype=np.uint8)
        arrpad[1:-1,1:-1,1:-1] = arr
        writer.add(arrpad, name)
    writer.close()

#####################################################
#load data from Tar-file and format into keras format
def load_and_format(fname_tar, cfg_model, load_tst_chunk = 0):
    #open TarFile
    reader = NpyTarReader(fname_tar)

    #if wanted load only a chunk to enable testing
    if load_tst_chunk is not 0:
        n_observations = load_tst_chunk
    else:
        n_observations = reader.elements()

    #keras format is (#observations x #channels x imgsize)
    th_format = (n_observations,) + cfg_model['dims'] + (cfg_model['n_channels'],)

    #create features & labels matrix
    dataset = np.zeros(th_format, dtype=np.float32)
    labels = []

    #iterate through all matrixes in tarfile and save as keras format numpy.ndarry and encoded labels
    for it, (x, name) in enumerate(reader):
        dataset[it] = x.astype(np.float32)
        labels.append(int(name.split('.')[0])-1)
        if len(labels) >= n_observations:
            break

    #return features & labels
    return dataset, labels

# use only if cross-validation not possible
def train_valid_split(dataset, labels, valid_size):
    data_train, data_valid, label_train, label_valid = train_test_split(dataset, labels, test_size= valid_size)
    return data_train, data_valid, label_train, label_valid

#shuffle dataset & labels
def shuffle(dataset, labels):
    dataset_sfl, labels_sfl = shuffle(dataset, labels)
    return dataset_sfl, labels_sfl

