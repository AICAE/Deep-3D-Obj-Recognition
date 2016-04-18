from __future__ import print_function
import numpy as np
import scipy.io
import os
import h5py

import logging


# loader needs a HDF file with a subgroup of name set_type
#  which holds a "labels_"+set_type and "features_"+set_type dataset


class Loader_hdf5:

    def __init__(self, fname,
                 batch_size=12 * 128, num_batches=None,
                 shuffle=False, valid_split=None,
                 mode="train"):

        logging.info("Loading dataset '{0}'".format(fname))

        try:
            self._openfile = h5py.File(fname)
        except:
            print("kein hdf5 file")
            raise IOError

        self._labels = self._openfile["train/labels_train"]

        self._features = self._openfile["train/features_train"]

        try:
            self._info = self._openfile["train/info_train"]
            self._has_rot = True
        except:
            self._has_rot = False

        self._labels_test = self._openfile["test/labels_test"]

        self._features_test = self._openfile["test/features_test"]

        # try:
        #     self._info_test = openfile["test/info_test"]
        #     self._has_rot = True
        # except:
        #     self._has_rot = False

        self._batch_size = batch_size
        self._num_batches = num_batches
        self._pos_train = 0
        self._pos_valid = 0
        self._pos_test = 0
        self._max_pos_train = None
        self._max_pos_valid = None
        self._min_pos_valid = None
        self._max_pos_test = None
        self._pos_train_indizes = None

        self.define_max_pos()

        self.sort_by_rotations()

        if shuffle is True:
            self.shuffle_data()

        if valid_split is not None:
            self._valid_size = valid_split
            self.validation_split()

        if mode == "train":
            self._mode = "train"
        elif mode == "valid":
            self._mode = "valid"
        elif mode == "test":
            self._mode = "test"

        logging.info("Done loading dataset.".format(fname))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._openfile.close()

    def define_max_pos(self):
        # if self._mode == "train":
        #     shape = self._labels_train.shape[0]
        # elif self._mode == "valid":
        #     shape = self._labels_valid.shape[0]
        # elif self._mode == "test":
        #     shape = self._labels_test.shape[0]

        shape = self._labels.len()
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_train = self._num_batches * self._batch_size
        else:
            self._max_pos_train = shape

        shape = self._labels_test.len()
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_test = self._num_batches * self._batch_size
        else:
            self._max_pos_test = shape

    def sort_by_rotations(self):
        self._pos_train_indizes = list(np.argsort(self._info[:, 1], axis=0))

    def shuffle_data(self):
        if self._has_rot is True:
            step_size = np.amax(self._info[:, 2]) - np.amin(self._info[:, 2]) + 1
        else:
            step_size = 1
        # Fisher-Yatest shuffle assuming that rotations of one obj are together
        for fy_i in range(self._labels.shape[0] - 1, 1 + step_size, -1 * step_size):
            fy_j = np.random.randint(1, int((fy_i + 1) / step_size) + 1) * step_size - 1
            if fy_j - step_size < 0:
                self._pos_train_indizes[fy_i:fy_i - step_size:-1], self._pos_train_indizes[fy_j::-1] =\
                    self._pos_train_indizes[fy_j::-1], self._pos_train_indizes[fy_i:fy_i - step_size:-1]
            else:
                self._pos_train_indizes[fy_i:fy_i - step_size:-1], self._pos_train_indizes[fy_j:fy_j - step_size:-1] =\
                    self._pos_train_indizes[fy_j:fy_j - step_size:-1], self._pos_train_indizes[fy_i:fy_i - step_size:-1]

    def validation_split(self):
        if self._has_rot is True:
            step_size = np.amax(self._info[:, 2]) - np.amin(self._info[:, 2]) + 1
        else:
            step_size = 1
        split_pos = int(int((self._labels.shape[0] / step_size) * (1 - self._valid_size)) * step_size)
        self._max_pos_valid = self._max_pos_train
        self._min_pos_valid = split_pos
        self._max_pos_train = split_pos

    def train_generator(self):
        self.change_mode("train")
        self.shuffle_data()
        self.validation_split()
        self._pos_train = 0
        while 1:

            features = self._features[sorted(self._pos_train_indizes[
                                      self._pos_train:self._pos_train + self._batch_size])]
            labels = self._labels[sorted(self._pos_train_indizes[
                self._pos_train:self._pos_train + self._batch_size])]

            self._pos_train += self._batch_size
            if self._pos_train > self._max_pos_train:
                self._pos_train = 0

            assert features.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(features.shape[0], self._batch_size, self._pos_train, self._max_pos_train)
            assert labels.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(labels.shape[0], self._batch_size, self._pos_train, self._max_pos_train)

            yield features, labels

    def return_num_train_samples(self):
        self.change_mode("train")
        return self._max_pos_train

    def valid_generator(self):
        self.change_mode("valid")
        self.shuffle_data()
        self.validation_split()
        while 1:

            features = self._features[sorted(self._pos_train_indizes[
                                      self._pos_valid:self._pos_valid + self._batch_size])]
            labels = self._labels[sorted(self._pos_train_indizes[
                                         self._pos_valid:self._pos_valid + self._batch_size])]

            self._pos_valid += self._batch_size
            if self._pos_valid > self._max_pos_valid:
                self._pos_valid = self._min_pos_valid

            assert features.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(features.shape[0], self._batch_size,
                           self._pos_valid - self._min_pos_valid,
                           self._max_pos_valid - self._min_pos_valid)
            assert labels.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(labels.shape[0], self._pos_valid - self._min_pos_valid,
                           self._batch_size, self._max_pos_valid - self._min_pos_valid)

            yield features, labels

    def return_num_valid_samples(self):
        self.change_mode("valid")
        return self._max_pos_valid

    def evaluate_generator(self):
        self.change_mode("test")
        while 1:
            features = self._features_test[self._pos_test:self._pos_test + self._batch_size, :, :, :, :]
            labels = self._labels_test[self._pos_test:self._pos_test + self._batch_size]

            self._pos_test += self._batch_size
            if self._pos_test > self._max_pos_test:
                self._pos_test = 0

            assert features.shape[0] == self._batch_size, \
                "in Evaluation Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(features.shape[0], self._batch_size, self._pos_test, self._max_pos_test)
            assert labels.shape[0] == self._batch_size, \
                "in Evaluation Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(labels.shape[0], self._batch_size, self._pos_test, self._max_pos_test)

            yield features, labels

    def return_num_evaluation_samples(self):
        self.change_mode("test")
        return self._max_pos_test

    ##-----------------------------------------------------------
    ## Activate this if loader should be iterable class
    ##
    # def __iter__(self):
    #     return self
    #
    # def __next__(self):
    #     return self.next()
    #
    # def next(self):
    #     if self._mode == "train":
    #         features = self._features_train[self._pos:self._pos + self._batch_size, :, :, :, :]
    #         labels = self._labels_train[self._pos:self._pos + self._batch_size]
    #     elif self._mode == "valid":
    #         features = self._features_valid[self._pos:self._pos + self._batch_size, :, :, :, :]
    #         labels = self._labels_valid[self._pos:self._pos + self._batch_size]
    #     elif self._mode == "test":
    #         features = self._features_test[self._pos:self._pos + self._batch_size, :, :, :, :]
    #         labels = self._labels_test[self._pos:self._pos + self._batch_size]
    #     else:
    #         features = None
    #         labels = None
    #
    #     self._pos += self._batch_size
    #     if self._pos > self._max_pos:
    #         raise StopIteration
    #
    #     return features, labels
    ##-----------------------------------------------------------

    def change_batch_size(self, batch_size):
        self._batch_size = batch_size

    def change_validation_size(self, valid_split):
        self._valid_size = valid_split
        self.validation_split()

    def change_mode(self, mode):
        self._mode = mode
        self.define_max_pos()
