from __future__ import print_function
import numpy as np
import npytar
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

import voxnet_tf_cfg

#load and format for Theano/keras
def load_and_format_th((fileTar, cfg, load_tst_chunk = 0):
    #open TarFile
    reader = npytar.NpyTarReader(fileTar)
    #if wanted load only a chunk to enable testing
    if load_tst_chunk is not 0:
        n_observations = load_tst_chunk
    else:
        n_observations = reader.elements()
    #theano format is (#observations x #channels x imgsize)
    tf_format = (n_observations,) + (cfg['n_channels'] + cfg['dims'],)
    #create features & labels matrix
    dataset = np.zeros(tf_format, dtype=np.float32)
    labels = []
    #iterate through all matrixes in tarfile and save as tensorflow format
    for it, (x, name) in enumerate(reader):
        dataset[it] = x.astype(np.float32)
        labels.append(int(name.split('.')[0])-1)
        if len(labels) >= n_observations:
            break
    #binaritize labels
    labels = (np.arange(cfg['n_classes']) == labels[:,None]).astype(np.float32)
    #return features & labels
    return dataset, labels

#load data from Tar-file and format into tensorflow format
def load_and_format_tf(fileTar, cfg, load_tst_chunk = 0):
    #open TarFile
    reader = npytar.NpyTarReader(fileTar)
    #if wanted load only a chunk to enable testing
    if load_tst_chunk is not 0:
        n_observations = load_tst_chunk
    else:
        n_observations = reader.elements()
    #tensorflow format is (#observations x imgsize x #channels)
    tf_format = (n_observations,) + cfg['dims'] + (cfg['n_channels'],)
    #create features & labels matrix
    dataset = np.zeros(tf_format, dtype=np.float32)
    labels = []
    #iterate through all matrixes in tarfile and save as tensorflow format
    for it, (x, name) in enumerate(reader):
        dataset[it] = x.astype(np.float32)
        labels.append(int(name.split('.')[0])-1)
        if len(labels) >= n_observations:
            break
    #binaritize labels
    labels = (np.arange(cfg['n_classes']) == labels[:,None]).astype(np.float32)
    #return features & labels
    return dataset, labels

# use only if now cross-validation possible
def train_valid_split(dataset, labels, valid_size):
    data_train, data_valid, label_train, label_valid = train_test_split(dataset, labels, test_size= valid_size)
    return data_train, data_valid, label_train, label_valid

#shuffle dataset & labels
def shuffle(dataset, labels):
    dataset_sfl, labels_sfl = shuffle(dataset, labels)
    return dataset_sfl, labels_sfl

