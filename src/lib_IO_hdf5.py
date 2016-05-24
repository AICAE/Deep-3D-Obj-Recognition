#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import scipy.io
import os
import h5py
import sys
from sklearn.preprocessing import label_binarize

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def save_dataset_as_hdf5(dirname_data, fname_save, class_name_to_id):
    """ 
    Args:
        dirname_data: string - the directory name where the volumetric data (from ModelNet40 - Princeton) is stored
        fname_save: string - the filename and including path that you want to use to save the hdf5 dataset
        class_name_to_id: dictonary - the classname to ID encoding file from the configfile

    Returns:
        Nothing - But it Saves a HDF5 Dataset at the given location with two subgroups "test" and "training" and the 
        corresponding files:
                test/features_test - Features Dataset of size [Num. Found TestFiles , 1 , 32 , 32 , 32]
                test/labels_test - Labels Dataset of size [Num. Found TestFiles ,]
                test/info_test - Info Dataset (LabelID, Object ID, Rotation ID) of size [Num. Found TestFiles , 3]
                train/features_train - Features Dataset of size [Num. Found trainFiles , 1 , 32 , 32 , 32]
                train/labels_train - Labels Dataset of size [Num. Found trainFiles ,]
                train/info_train - Info Dataset (LabelID, Object ID, Rotation ID) of size [Num. Found trainFiles , 3]

    Procedure:
        1.) Scan Directory for all .mat files that match the requirements in class_name_to_id and count training & test
            files
        2.) Create six Numpy Arrays (feat_test, feat-train, label_test, label_train, info_test, info_train) that are
            big enough to store all information
        3.) iterate through all .mat files load and save data in features, extract the classID, objectID, rotationID
            from the filename of the .mat file and save the information in the labels and info matrix
        4.) create HDF5 file with corresponding data structure and save numpy arrays in corresponding dataset

    Examples:
        save-dataset_as_hdf5("data/volumetric_data","data/modelnet10.hdf5",class_name_to_id_modelnet10)

    """
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

                # some classes carry _ in name
                try:
                    int(fname[pos1 + 1])
                except ValueError:
                    pos1 = fname.find('_', pos1 + 1)

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

    logging.info("found {0} train files and {1} test files".format(train_size, test_size))

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
                # some classes carry _ in anme
                try:
                    int(fname[pos1 + 1])
                except ValueError:
                    pos1 = fname.find('_', pos1 + 1)

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

    logging.info("extracted {0} train datasets and {1} test datasets".format(lab_train.shape[0], lab_test.shape[0]))

    # create hdf5 dataset storage and iterate through all files and save them. FileSize Reason
    openfile = h5py.File(fname_save, 'w')
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
    logging.info("saved datasets as hdf at {0}".format(fname_save))


class Loader_hdf5_Convert_Np:
    """
        This is the loader class that loads the hdf5 file and converts it to a numpy array.
        All Operations are than performed on the Numpy Arrays.

        This class needs more RAM but will be a lot faster.

    """

    def __init__(self, fname,
                 batch_size=128,
                 num_batches=None,
                 has_rot=False,
                 shuffle=False,
                 valid_split=None,
                 mode="train"):
        """

        Args:
            fname: string -> HDF5 file - filename of the Dataset
            batch_size: Integer - The batch_size that should be  returned by the generators
            num_batches: Integer - the number of batches that should be returned by the generators - Integer
            has_rot: Boolean - with this option you can decide whether the class should search for rotations or not
                                rotations can only be found if there is an info file
            shuffle: Boolean - option to mark if set should be shuffled
            valid_split: float in range (0,1) or None - if a float is given the class will spplit the training set into
                                                        a training and validation set
            mode: "train, valid, test" - mode to show which type of set si returned currently

        Returns:
            self: ClassObject

        Procedure:
            1.) Load the HDF5 file
            2.) load all Dataset and save them in Numpy Arrays
            3.) Close HDF5 file
            4.) Sort by Rotations
            5.) Shuffle Data
            6.) Create Validation Set
            7.) Define Size of Train, Validation & Test Array

        """

        logging.info("Loading dataset '{0}'".format(fname))
        openfile = h5py.File(fname)

        lab = openfile["train/labels_train"]
        self._labels = np.zeros(lab.shape, dtype=np.uint32)
        lab.read_direct(self._labels)

        feat = openfile["train/features_train"]
        self._features = np.zeros(feat.shape, dtype=np.uint8)
        feat.read_direct(self._features)

        logging.info("Found Training Data of shape: {0}".format(feat.shape))

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

        logging.info("Found Test Data of shape: {0}".format(feat_test.shape))

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

        self._classes = np.unique(self._labels)
        self._num_classes = self._classes.shape[0]
        logging.info("found {0} classes: \n {1}".format(self._num_classes, self._classes))

        self.sort_by_rotations()

        if shuffle is True:
            self.shuffle_data()

        if isinstance(valid_split,float) and valid_split > 0.0:
            self._valid_size = valid_split
            self.validation_split()
        else:
            self._valid_size = 0.0
            self._features_train = self._features
            self._labels_train = self._labels

        self.define_max_pos()

        self._features = None
        self._labels = None

        logging.info("Done loading dataset.".format(fname))

    def sort_by_rotations(self):
        """

        This function sorts all arrays based on the Object ID of the Info Array

        Procedure:
            1.) Define Sort Indizes based on Info Array
            2.) Sort all arrays using Sort Indizes

        """
        if self._has_rot:
            sort_scheme = np.argsort(self._info[:, 1], axis=0)
            self._info = self._info[sort_scheme]
            self._features = self._features[sort_scheme]
            self._labels = self._labels[sort_scheme]

    def shuffle_data(self):
        """

        This Function Shuffles the Arrays, if has_rot is enabled it will do a batch shuffle and shuffle objects with
        the same ObjectID from Info(:,1) together. Otherwise it performs normal elementwise shuffle

        Procedure:
            1.) Figure out number of Rotations if has_rot is set, other wise elementwise
            2.) Perform Fisher-Yates shuffle directly on Array

        """
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
        """

        This functions performs a Validation split on the feature and label array, it will therefore split the array
        into fou new arrays features_train labels_train features_valid labels_valid. The size of the validation Array
        is determined by self_valid_size. If rotations are present the split will be set after the last Rotation of
        one Object

        Procedure:
            1.) Figure out number of Rotations if has_rot is set, other wise elementwise
            2.) Determine Split Position
            3.) Split features and labels into four new arrays

        """
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
        """

        This functions determines the size of all arrays and sets the point where the iteration of the generator
        will be reset to the start, based on if there is a number of batches given.

        Procedure:
            1.) Determine Shape of array
            2.) if number of batches if given and number of batches times batch size is smaller than shape set
                maximum iteration position to number of batches times batch size otherwise to length of array
                minus the modulo of the length of array, to ensure only full batches are loaded


        """
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
            self._max_pos_train = shape - shape % self._batch_size

        shape = self._labels_valid.shape[0]
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_valid = self._num_batches * self._batch_size
        else:
            self._max_pos_valid = shape - shape % self._batch_size

        shape = self._labels_test.shape[0]
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_test = self._num_batches * self._batch_size
        else:
            self._max_pos_test = shape - shape % self._batch_size

    def train_generator(self):
        """

        This it the generator for the training data.

        Yields:
            features - Array of size [batch_size, 1 , 32 , 32 , 32]
            labels - Array of size [batch-size,]


        Procedure:
            1.) Reset position of iterator and redefine maximum position of iterator
            x.) indefinetly extract the features & labels at the current position of the iterator, of size batch_size
                if the maximum position of the iterator is reached reset iterator to 0

        """
        logging.info("Initialize Train Generator")
        self.define_max_pos()
        self._pos_train = 0
        while 1:
            features = self._features_train[self._pos_train:self._pos_train + self._batch_size]
            labels = self._labels_train[self._pos_train:self._pos_train + self._batch_size]

            labels_binary = label_binarize(labels, self._classes)

            self._pos_train += self._batch_size
            if self._pos_train >= self._max_pos_train:
                self._pos_train = 0

            assert features.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(features.shape[0], self._batch_size, self._pos_train, self._max_pos_train)
            assert labels_binary.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(labels.shape[0], self._batch_size, self._pos_train, self._max_pos_train)

            yield features, labels_binary

    def return_num_train_samples(self):
        """

        Returns:
            _max_pos_train = Integer - Is the number of  training samples which will be returned
            by the generator in one run

        """
        self.define_max_pos()
        return self._max_pos_train

    def valid_generator(self):
        """

        This it the generator for the validation data.

        Yields:
            features - Array of size [batch_size, 1 , 32 , 32 , 32]
            labels - Array of size [batch-size,]


        Procedure:
            1.) Reset position of iterator and redefine maximum position of iterator
            x.) indefinetly extract the features & labels at the current position of the iterator, of size batch_size
                if the maximum position of the iterator is reached reset iterator to 0

        """
        if self._valid_size > 0.0:
            return self._valid_generator()
        else:
            return None

    def _valid_generator(self):
        logging.info("Initialize Valid Generator")
        self.define_max_pos()
        self._pos_valid = 0
        while 1:

            features = self._features_valid[self._pos_valid:self._pos_valid + self._batch_size]
            labels = self._labels_valid[self._pos_valid:self._pos_valid + self._batch_size]

            labels_binary = label_binarize(labels, self._classes)

            self._pos_valid += self._batch_size
            if self._pos_valid >= self._max_pos_valid:
                self._pos_valid = 0

            assert features.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(features.shape[0], self._batch_size, self._pos_valid, self._max_pos_valid)
            assert labels_binary.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(labels_binary.shape[0], self._pos_valid, self._batch_size, self._max_pos_valid)

            yield features, labels_binary

    def return_num_valid_samples(self):
        """

        Returns:
            _max_pos_valid = Integer - Is the number of  validation  samples which will be returned
            by the generator in one run

        """
        if self._valid_size > 0.0:
            self.define_max_pos()
            return self._max_pos_valid
        else:
            return None


    def evaluate_generator(self):
        """

        This it the generator for the test data.

        Yields:
            features - Array of size [batch_size, 1 , 32 , 32 , 32]
            labels - Array of size [batch-size,]


        Procedure:
            1.) Reset position of iterator and redefine maximum position of iterator
            x.) indefinetly extract the features & labels at the current position of the iterator, of size batch_size
                if the maximum position of the iterator is reached reset iterator to 0

        """
        logging.info("Initialize Evaluation Generator")
        self.define_max_pos()
        self._pos_test = 0
        while 1:
            features = self._features_test[self._pos_test:self._pos_test + self._batch_size]
            labels = self._labels_test[self._pos_test:self._pos_test + self._batch_size]

            labels_binary = label_binarize(labels, self._classes)

            self._pos_test += self._batch_size
            if self._pos_test >= self._max_pos_test:
                self._pos_test = 0

            assert features.shape[0] == self._batch_size, \
                "in Evaluation Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(features.shape[0], self._batch_size, self._pos_test, self._max_pos_test)
            assert labels_binary.shape[0] == self._batch_size, \
                "in Evaluation Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(labels_binary.shape[0], self._batch_size, self._pos_test, self._max_pos_test)

            yield features, labels_binary

    def return_num_evaluation_samples(self):
        """

        Returns:
            _max_pos_test = Integer - Is the number of Evaluation samples which will be returned
            by the generator in one run.

        """
        self.define_max_pos()
        return self._max_pos_test

    def return_valid_set(self):
        """

        Returns:

        """
        return self._features_valid, self._labels_valid

    def change_batch_size(self, batch_size):
        """

        Args:
            batch_size:

        Returns:

        """
        self._batch_size = batch_size

    def change_validation_size(self, valid_split):
        """

        Args:
            valid_split:

        Returns:

        """
        self._valid_size = valid_split
        self.validation_split()

    def return_nb_classes(self):
        """

        Returns: number of classes in dataset as int

        """
        return self._num_classes

    # -----------------------------------------------------------
    # Activate this if loader should be iterable class
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
    # -----------------------------------------------------------


class Loader_hdf5:
    """

    This loader only opens the HDF5 file and only loads the pieces from the HDF5 file that are passed during iteration
    of the generator. All shuffle, split, etc. options are performed on Indizes solely.

    This class should use less RAM but will be a lot slower

    """

    def __init__(self, fname,
                 batch_size=128,
                 num_batches=None,
                 has_rot=False,
                 shuffle=False,
                 valid_split=None,
                 ):
        """

        Args:
            fname: string -> HDF5 file - filename of the Dataset
            batch_size: Integer - The batch_size that should be  returned by the generators
            num_batches: Integer - the number of batches that should be returned by the generators - Integer
            has_rot: Boolean - with this option you can decide whether the class should search for rotations or not
                                rotations can only be found if there is an info file
            shuffle: Boolean - option to mark if set should be shuffled
            valid_split: float in range (0,1) or None - if a float is given the class will spplit the training set into
                                                        a training and validation set
            mode: "train, valid, test" - mode to show which type of set si returned currently

        Returns:
            self: ClassObject

        Procedure:
            1.) Load the HDF5 file
            2.) Define Size of Arrays
            3.) Sort by Rotations
            4.) Shuffle Data
            5.) Create Validation Set

        """
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

        self._classes = np.unique(self._labels)
        self._num_classes = self._classes.shape[0]
        logging.info("found {0} classes : \n {1}".format(self._num_classes, self._classes))

        self.sort_by_rotations()

        if shuffle is True:
            self.shuffle_data()

        if valid_split is not None:
            self._valid_size = valid_split
            self.validation_split()

        logging.info("loaderHDF5 initialized.".format(fname))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._openfile.close()

    def define_max_pos(self):
        """

        This functions determines the size of all arrays and sets the point where the iteration of the generator
        will be reset to the start, based on if there is a number of batches given.

        Procedure:
            1.) Determine Shape of array
            2.) if number of batches if given and number of batches times batch size is smaller than shape set
                maximum iteration position to number of batches times batch size otherwise to length of array

        """
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
            self._max_pos_train = shape - shape % self._batch_size

        shape = self._labels_test.len()
        if self._num_batches is not None and self._num_batches * self._batch_size < shape:
            self._max_pos_test = self._num_batches * self._batch_size
        else:
            self._max_pos_test = shape - shape % self._batch_size

    def sort_by_rotations(self):
        """

        This function sorts all Indizes of the Arrays based on the Object ID of the Info Array

        Procedure:
            1.) Define Sort Indizes based on Info Array

        """
        self._pos_train_indizes = list(np.argsort(self._info[:, 1], axis=0))

    def shuffle_data(self):
        """

        This Function Shuffles the Indizes of the Arrays, if has_rot is enabled it will do a batch shuffle
        and shuffle Indizes of objects with the same ObjectID from Info(:,1) together.
        Otherwise it performs normal elementwise shuffle

        Procedure:
            1.) Figure out number of Rotations if has_rot is set, other wise elementwise
            2.) Perform Fisher-Yates shuffle on Indizes

        """
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
                    self._pos_train_indizes[
                        fy_j:fy_j - self._num_rot:-1], self._pos_train_indizes[fy_i:fy_i - self._num_rot:-1]

    def validation_split(self):
        """

        This functions defines the split position for the feature and label array. The size of the validation Array
        is determined by self_valid_size. If rotations are present the split will be set after the last Rotation of
        one Object

        Procedure:
            1.) Figure out number of Rotations if has_rot is set, other wise elementwise
            2.) Determine Split Position and set min and max values for train and valid accordingly

        """
        if self._has_rot is True:
            self._num_rot = np.amax(self._info[:, 2]) - np.amin(self._info[:, 2]) + 1
        else:
            self._num_rot = 1
        split_pos = int(int((self._labels.shape[0] / self._num_rot) * (1 - self._valid_size)) * self._num_rot)
        self._max_pos_valid = self._max_pos_train
        self._min_pos_valid = split_pos
        self._max_pos_train = split_pos

    def train_generator(self):
        """
        This it the generator for the training data.

        Yields:
            features - Array of size [batch_size, 1 , 32 , 32 , 32]
            labels - Array of size [batch-size,]


        Procedure:
            1.) Reset position of iterator and redefine maximum position of iterator
            x.1) indefinetly extract the features & labels at the current position of the iterator for the indizes,
                 by loading the samples for the according Indizes form the HDF5 file into a Numpy Array and yield it
            x.2) if the maximum position of the iterator is reached reset iterator to 0 and shuffle Indizes before next
                 Iteration cycle

        """
        self.define_max_pos()
        self.shuffle_data()
        self.validation_split()
        self._pos_train = 0
        #shp_features = list(self._features.shape[1:])
        # shp_features.insert(0,self._batch_size)
        #features_tmp = np.zeros(shp_features, dtype=np.uint8)
        #shp_labels = list(self._labels.shape[1:])
        # shp_labels.insert(0,self._batch_size)
        #labels_tmp = np.zeros(shp_labels, dtype=np.uint32)
        while 1:

            features_tmp = self._features[sorted(self._pos_train_indizes[
                self._pos_train:self._pos_train + self._batch_size])]
            labels_tmp = self._labels[sorted(self._pos_train_indizes[
                self._pos_train:self._pos_train + self._batch_size])]

            labels_binary = label_binarize(labels_tmp, self._classes)

            self._pos_train += self._batch_size
            if self._pos_train >= self._max_pos_train:
                self._pos_train = 0
                self.shuffle_data()

            assert features_tmp.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(features_tmp.shape[0], self._batch_size, self._pos_train, self._max_pos_train)
            assert labels_binary.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(labels_binary.shape[0], self._batch_size, self._pos_train, self._max_pos_train)

            yield features_tmp, labels_binary

    def return_num_train_samples(self):
        """

        Returns:
            _max_pos_test = Integer - Is the number of Training samples which will be returned
            by the generator in one run.

        """
        self.define_max_pos()
        return self._max_pos_train

    def valid_generator(self):
        """
        This it the generator for the validation data.

        Yields:
            features - Array of size [batch_size, 1 , 32 , 32 , 32]
            labels - Array of size [batch-size,]


        Procedure:
            1.) Reset position of iterator and redefine maximum position of iterator
            x.1) indefinetly extract the features & labels at the current position of the iterator for the indizes,
                 by loading the samples for the according Indizes form the HDF5 file into a Numpy Array and yield it
            x.2) if the maximum position of the iterator is reached reset iterator to 0

        """
        self.define_max_pos()
        self.shuffle_data()
        self.validation_split()
        #shp_features = list(self._features.shape[1:])
        # shp_features.insert(0,self._batch_size)
        #features_tmp = np.zeros(shp_features, dtype=np.uint8)
        #shp_labels = list(self._labels.shape[1:])
        # shp_labels.insert(0,self._batch_size)
        #labels_tmp = np.zeros(shp_labels, dtype=np.uint32)
        while 1:
            features_tmp = self._features[sorted(self._pos_train_indizes[
                self._pos_valid:self._pos_valid + self._batch_size])]

            labels_tmp = self._labels[sorted(self._pos_train_indizes[
                self._pos_valid:self._pos_valid + self._batch_size])]

            labels_binary = label_binarize(labels_tmp, self._classes)

            self._pos_valid += self._batch_size
            if self._pos_valid >= self._max_pos_valid:
                self._pos_valid = self._min_pos_valid

            assert features_tmp.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(features_tmp.shape[0], self._batch_size,
                       self._pos_valid - self._min_pos_valid,
                       self._max_pos_valid - self._min_pos_valid)
            assert labels_binary.shape[0] == self._batch_size,\
                "in Valid Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(labels_binary.shape[0], self._pos_valid - self._min_pos_valid,
                       self._batch_size, self._max_pos_valid - self._min_pos_valid)

            yield features_tmp, labels_binary

    def return_num_valid_samples(self):
        """

        Returns:
            _max_pos_va =lid Integer - Is the number of validation samples which will be returned
            by the generator in one run.

        """
        self.define_max_pos()
        return self._max_pos_valid

    def evaluate_generator(self):
        """
        This it the generator for the evaluation data.

        Yields:
            features - Array of size [batch_size, 1 , 32 , 32 , 32]
            labels - Array of size [batch-size,]


        Procedure:
            1.) Reset position of iterator and redefine maximum position of iterator
            x.1) indefinetly extract the features & labels at the current position of the iterator for the batch_size,
                 by loading the samples for the according Indizes form the HDF5 file into a Numpy Array and yield it
            x.2) if the maximum position of the iterator is reached reset iterator to 0

        """
        self.define_max_pos()
        #shp_features = list(self._features_test.shape[1:])
        # shp_features.insert(0,self._batch_size)
        #features_tmp = np.zeros(shp_features, dtype=np.uint8)
        #shp_labels = list(self._labels_test.shape[1:])
        # shp_labels.insert(0,self._batch_size)
        #labels_tmp = np.zeros(shp_labels, dtype=np.uint32)
        while 1:
            features_tmp = self._features_test[self._pos_test:self._pos_test + self._batch_size]

            labels_tmp = self._labels_test[self._pos_test:self._pos_test + self._batch_size]

            labels_binary = label_binarize(labels_tmp, self._classes)

            self._pos_test += self._batch_size
            if self._pos_test >= self._max_pos_test:
                self._pos_test = 0

            assert features_tmp.shape[0] == self._batch_size, \
                "in Evaluation Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(features_tmp.shape[0], self._batch_size, self._pos_test, self._max_pos_test)
            assert labels_binary.shape[0] == self._batch_size, \
                "in Evaluation Generator features of wrong shape is {0} should be {1} at pos {2} of max_pos {3}".\
                format(labels_binary.shape[0], self._batch_size, self._pos_test, self._max_pos_test)

            yield features_tmp, labels_binary

    def return_num_evaluation_samples(self):
        """

        Returns:
            _max_pos_test = Integer - Is the number of Evaluation samples which will be returned
            by the generator in one run.

        """
        self.define_max_pos()
        return self._max_pos_test

    # -----------------------------------------------------------
    # Activate this if loader should be iterable class
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
    # -----------------------------------------------------------

    def change_batch_size(self, batch_size):
        self._batch_size = batch_size

    def change_validation_size(self, valid_split):
        self._valid_size = valid_split
        self.validation_split()

    def return_nb_classes(self):
        return self._num_classes
