from __future__ import print_function
import numpy as np
import scipy.io
import os
import h5py

import logging


def save_dataset_as_hdf5(dirname_data, fname_save, class_name_to_id):

    classnames = set(class_name_to_id.keys())

    # find out how many files for test and train exist. Speed Reasons
    train_size = 0
    test_size = 0
    for dirpaths, dirs, fnames in os.walk(dirname_data):
        for fname in fnames:
            if fname.endswith('.mat'):
                pos1 = fname.find('_')
                # the correct files have a _ before the rotation
                if pos1 is -1:
                    continue

                # check if loaded class is one of the required classes
                classname = fname[:pos1]
                if classname not in classnames:
                    continue

                pos2 = dirpaths.rfind('/')
                set_type = dirpaths[pos2 + 1:]
                if set_type == 'train':
                    train_size += 1
                elif set_type == 'test':
                    test_size += 1

    print("found {0} train files and {1} test files".format(train_size, test_size))

    feat_train = np.zeros([train_size, 1, 32, 32, 32], dtype=np.uint8)
    lab_train = np.zeros([train_size, ], dtype=np.uint32)
    info_train = np.zeros([train_size, 3], dtype=np.uint32)
    feat_test = np.zeros([test_size, 1, 32, 32, 32], dtype=np.uint8)
    lab_test = np.zeros([test_size, ], dtype=np.uint32)
    info_test = np.zeros([test_size, 3], dtype=np.uint32)

    train_it = 0
    test_it = 0
    classnames = set(class_name_to_id.keys())

    for dirpaths, dirs, fnames in os.walk(dirname_data):
        for fname in fnames:
            if fname.endswith('.mat'):
                pos1 = fname.find('_')

                if pos1 is -1:
                    continue

                # check if loaded class is one of the required classes
                classname = fname[:pos1]
                if classname not in classnames:
                    continue

                # find set_type info
                pos2 = dirpaths.rfind('/')
                set_type = dirpaths[pos2 + 1:]

                # find obj_ID info
                pos3 = fname.find('_', pos1 + 1)
                obj_id = int(fname[pos1 + 1:pos3])

                # find rotation_Id info
                pos4 = fname.find('.', pos3 + 1)
                rot_id = int(fname[pos3 + 1:pos4])

                # encode class and add to labels
                label = int(class_name_to_id[classname])

                # load mat, reshape to 32x32x32 array and add to save to .npy file
                arr = scipy.io.loadmat(os.path.join(dirpaths, fname))['instance'].astype(np.uint8)

                if set_type == "train":
                    feat_train[train_it, 0, 1:-1, 1:-1, 1:-1] = arr
                    lab_train[train_it] = label
                    info_train[train_it, :] = [label, obj_id, rot_id]
                    train_it += 1
                elif set_type == "test":
                    feat_test[test_it, 0, 1:-1, 1:-1, 1:-1] = arr
                    lab_test[test_it] = label
                    info_test[test_it, :] = [label, obj_id, rot_id]
                    test_it += 1

    print("found {0} train datasets and {1} test datasets".format(lab_train.shape, lab_test.shape))

    # create hdf5 dataset storage and iterate through all files and save them. FileSize Reason
    openfile = h5py.File(fname_save, "w")
    train = openfile.create_group("train")
    train.create_dataset("features_train", [train_size, 1, 32, 32, 32],
                         dtype=np.uint8,
                         chunks=True,
                         compression="gzip",
                         data=feat_train)
    train.create_dataset("labels_train", [train_size, ],
                         dtype=np.uint32,
                         chunks=True,
                         compression="gzip",
                         data=lab_train)
    train.create_dataset("info_train", [train_size, 3],
                         dtype=np.uint32,
                         chunks=True,
                         compression="gzip",
                         data=info_train)
    test = openfile.create_group("test")
    test.create_dataset("features_test", [test_size, 1, 32, 32, 32],
                        dtype=np.uint8,
                        chunks=True,
                        compression="gzip",
                        data=feat_test)
    test.create_dataset("labels_test", [test_size, ],
                        dtype=np.uint32,
                        chunks=True,
                        compression="gzip",
                        data=lab_test)
    test.create_dataset("info_test", [test_size, 3],
                        dtype=np.uint32,
                        chunks=True,
                        compression="gzip",
                        data=info_test)
    openfile.close()

# loader needs a HDF file with a subgroup of name set_type
#  which holds a "labels_"+set_type and "features_"+set_type dataset


class Loader_hdf5_Convert_Np:

    def __init__(self, fname,
                 batch_size=128, num_batches=None,
                 has_rot = False,
                 shuffle=False, valid_split=None,
                 mode="train"):

        logging.info("Loading dataset '{0}'".format(fname))
        openfile = h5py.File(fname)

        lab = openfile["train/labels_train"]
        self._labels = np.zeros(lab.shape, dtype=np.uint32)
        lab.read_direct(self._labels)

        feat = openfile["train/features_train"]
        self._features = np.zeros(feat.shape, dtype=np.uint8)
        feat.read_direct(self._features)

        try:
            info = openfile["train/info_train"]
            self._info = np.zeros(info.shape, dtype=np.uint32)
            info.read_direct(self._info)
            if has_rot is False:
                self._has_rot = False
            else:
                self._has_rot = True
        except IOError:
            self._has_rot = False

        lab_test = openfile["test/labels_test"]
        self._labels_test = np.zeros(lab_test.shape, dtype=np.uint32)
        lab_test.read_direct(self._labels_test)

        feat_test = openfile["test/features_test"]
        self._features_test = np.zeros(feat_test.shape, dtype=np.uint8)
        feat_test.read_direct(self._features_test)

        try:
            info_test = openfile["test/info_test"]
            self._info_test = np.zeros(info_test.shape, dtype=np.uint32)
            info_test.read_direct(self._info_test)
            if has_rot is False:
                self._has_rot = False
            else:
                self._has_rot = True
        except IOError:
            self._has_rot = False

        openfile.close()

        self._batch_size = batch_size
        self._num_batches = num_batches
        self._pos_train = 0
        self._pos_valid = 0
        self._pos_test = 0
        self._max_pos_train = None
        self._max_pos_valid = None
        self._max_pos_test = None
        self._num_rot = None

        self.sort_by_rotations()

        if shuffle is True:
            self.shuffle_data()

        if valid_split is not None:
            self._valid_size = valid_split
            self.validation_split()
        else:
            self._features_train = self._features
            self._labels_train = self._labels

        if mode == "train":
            self._mode = "train"
        elif mode == "valid":
            self._mode = "valid"
        elif mode == "test":
            self._mode = "test"

        self.define_max_pos()

        self._features = None
        self._labels = None

        logging.info("Done loading dataset.".format(fname))

    def sort_by_rotations(self):
        sort_scheme = np.argsort(self._info[:, 1], axis=0)
        self._info = self._info[sort_scheme]
        self._features = self._features[sort_scheme]
        self._labels = self._labels[sort_scheme]

    def shuffle_data(self):
        if self._has_rot is True:
            self._num_rot = np.amax(self._info[:, 2]) - np.amin(self._info[:, 2]) + 1
        else:
            self._num_rot = 1
        # Fisher-Yatest shuffle assuming that rotations of one obj are together
        for fy_i in range(self._labels.shape[0] - 1, 1 + self._num_rot, -1 * self._num_rot):
            fy_j = np.random.randint(1, int((fy_i + 1) / self._num_rot) + 1) * self._num_rot - 1
            if fy_j - self._num_rot < 0:
                self._features[fy_i:fy_i - self._num_rot:-1], self._features[fy_j::-1] =\
                    self._features[fy_j::-1], self._features[fy_i:fy_i - self._num_rot:-1].copy()
                self._labels[fy_i:fy_i - self._num_rot:-1], self._labels[fy_j::-1] =\
                    self._labels[fy_j::-1], self._labels[fy_i:fy_i - self._num_rot:-1].copy()
                self._info[fy_i:fy_i - self._num_rot:-1], self._info[fy_j::-1] =\
                    self._info[fy_j::-1], self._info[fy_i:fy_i - self._num_rot:-1].copy()
            else:
                self._features[fy_i:fy_i - self._num_rot:-1], self._features[fy_j:fy_j - self._num_rot:-1] =\
                    self._features[fy_j:fy_j - self._num_rot:-1], self._features[fy_i:fy_i - self._num_rot:-1].copy()
                self._labels[fy_i:fy_i - self._num_rot:-1], self._labels[fy_j:fy_j - self._num_rot:-1] =\
                    self._labels[fy_j:fy_j - self._num_rot:-1], self._labels[fy_i:fy_i - self._num_rot:-1].copy()
                self._info[fy_i:fy_i - self._num_rot:-1], self._info[fy_j:fy_j - self._num_rot:-1] =\
                    self._info[fy_j:fy_j - self._num_rot:-1], self._info[fy_i:fy_i - self._num_rot:-1].copy()

    def validation_split(self):
        if self._has_rot is True:
            self._num_rot = np.amax(self._info[:, 2]) - np.amin(self._info[:, 2]) + 1
        else:
            self._num_rot = 1
        split_pos = int(int((self._labels.shape[0] / self._num_rot) * (1 - self._valid_size)) * self._num_rot)
        self._features_train = self._features[:split_pos]
        self._labels_train = self._labels[:split_pos]
        self._features_valid = self._features[split_pos:]
        self._labels_valid = self._labels[split_pos:]

    def define_max_pos(self):
        # if self._mode == "train":
        #     shape = self._labels_train.shape[0]
        # elif self._mode == "valid":
        #     shape = self._labels_valid.shape[0]
        # elif self._mode == "test":
        #     shape = self._labels_test.shape[0]

        shape = self._labels_train.shape[0]
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_train = self._num_batches * self._batch_size
        else:
            self._max_pos_train = shape

        shape = self._labels_valid.shape[0]
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_valid = self._num_batches * self._batch_size
        else:
            self._max_pos_valid = shape

        shape = self._labels_test.shape[0]
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_test = self._num_batches * self._batch_size
        else:
            self._max_pos_test = shape

    def train_generator(self):
        logging.info("Initialize Train Generator")
        self.change_mode("train")
        self._pos_train = 0
        while 1:
            features = self._features_train[self._pos_train:self._pos_train + self._batch_size]
            labels = self._labels_train[self._pos_train:self._pos_train + self._batch_size]

            self._pos_train += self._batch_size
            if self._pos_train >= self._max_pos_train:
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
        logging.info("Initialize Valid Generator")
        self.change_mode("valid")
        self._pos_valid = 0
        while 1:

            features = self._features_valid[self._pos_valid:self._pos_valid + self._batch_size]
            labels = self._labels_valid[self._pos_valid:self._pos_valid + self._batch_size]

            self._pos_valid += self._batch_size
            if self._pos_valid >= self._max_pos_valid:
                self._pos_valid = 0

            assert features.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(features.shape[0], self._batch_size, self._pos_valid, self._max_pos_valid)
            assert labels.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                    format(labels.shape[0], self._pos_valid, self._batch_size, self._max_pos_valid)

            yield features, labels

    def return_num_valid_samples(self):
        self.change_mode("valid")
        return self._max_pos_valid

    def evaluate_generator(self):
        logging.info("Initialize Evaluation Generator")
        self.change_mode("test")
        self._pos_test = 0
        while 1:
            features = self._features_test[self._pos_test:self._pos_test + self._batch_size]
            labels = self._labels_test[self._pos_test:self._pos_test + self._batch_size]

            self._pos_test += self._batch_size
            if self._pos_test >= self._max_pos_test:
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

    def change_mode(self, mode):
        self._mode = mode
        self.define_max_pos()

    def return_valid_set(self):
        return self._features_valid, self._labels_valid

    def change_batch_size(self, batch_size):
        self._batch_size = batch_size

    def change_validation_size(self, valid_split):
        self._valid_size = valid_split
        self.validation_split()

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

class Loader_hdf5:

    def __init__(self, fname,
                 batch_size=12 * 128, num_batches=None,
                 has_rot = False,
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
            if has_rot is False:
                self._has_rot = False
            else:
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
        self._num_rot = None

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
            self._num_rot = np.amax(self._info[:, 2]) - np.amin(self._info[:, 2]) + 1
        else:
            self._num_rot = 1
        # Fisher-Yatest shuffle assuming that rotations of one obj are together
        for fy_i in range(self._labels.shape[0] - 1, 1 + self._num_rot, -1 * self._num_rot):
            fy_j = np.random.randint(1, int((fy_i + 1) / self._num_rot) + 1) * self._num_rot - 1
            if fy_j - self._num_rot < 0:
                self._pos_train_indizes[fy_i:fy_i - self._num_rot:-1], self._pos_train_indizes[fy_j::-1] =\
                    self._pos_train_indizes[fy_j::-1], self._pos_train_indizes[fy_i:fy_i - self._num_rot:-1]
            else:
                self._pos_train_indizes[fy_i:fy_i - self._num_rot:-1], self._pos_train_indizes[fy_j:fy_j - self._num_rot:-1] =\
                    self._pos_train_indizes[fy_j:fy_j - self._num_rot:-1], self._pos_train_indizes[fy_i:fy_i - self._num_rot:-1]

    def validation_split(self):
        if self._has_rot is True:
            self._num_rot = np.amax(self._info[:, 2]) - np.amin(self._info[:, 2]) + 1
        else:
            self._num_rot = 1
        split_pos = int(int((self._labels.shape[0] / self._num_rot) * (1 - self._valid_size)) * self._num_rot)
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
            if self._pos_train >= self._max_pos_train:
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
            if self._pos_valid >= self._max_pos_valid:
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
            features = self._features_test[self._pos_test:self._pos_test + self._batch_size]
            labels = self._labels_test[self._pos_test:self._pos_test + self._batch_size]

            self._pos_test += self._batch_size
            if self._pos_test >= self._max_pos_test:
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