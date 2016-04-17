#!/usr/bin/python3

import keras
assert keras.__version__ == "1.0.0", "keras version not supported"

from keras import backend as K

from keras.models import Sequential

from keras.layers import Convolution3D, MaxPooling3D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.engine.training import batch_shuffle

from keras.optimizers import SGD

from Source.voxnet_keras import lib_IO_hdf5

import logging


class model_vt (object):

    # TODO l2(0.001) for EVERY single layer required or sufficient if applied to last layer ?
    def __init__(self):
        """initiate Model according to voxnet paper"""
        # Stochastic Gradient Decent (SGD) with momentum
        # lr=0.01 for LiDar dataset
        # lr=0.001 for other datasets
        # decay of 0.00016667 approx the same as learning schedule (0:0.001,60000:0.0001,600000:0.00001)
        self._optimizer = SGD(lr=0.001, momentum=0.9, decay=0.00016667, nesterov=False)

        # init model
        self._mdl = Sequential()

        # convolution 1
        self._mdl.add(Convolution3D(input_shape=(1, 32, 32, 32),
                                    nb_filter=32,
                                    kernel_dim1=5,
                                    kernel_dim2=5,
                                    kernel_dim3=5,
                                    init='normal',  # TODO
                                    weights=None,  # TODO
                                    border_mode='valid',
                                    subsample=(2, 2, 2),
                                    dim_ordering='th',
                                    W_regularizer=l2(0.001),
                                    b_regularizer=l2(0.001),
                                    activity_regularizer=None,
                                    W_constraint=None,
                                    b_constraint=None))

        logging.debug("Layer1:Conv3D shape={0}".format(self._mdl.output_shape))
        self._mdl.add(Activation(LeakyReLU(alpha=0.1)))

        # dropout 1
        self._mdl.add(Dropout(p=0.2))

        # convolution 2
        self._mdl.add(Convolution3D(nb_filter=32,
                                    kernel_dim1=3,
                                    kernel_dim2=3,
                                    kernel_dim3=3,
                                    init='normal',  # TODO
                                    weights=None,  # TODO
                                    border_mode='valid',
                                    subsample=(1, 1, 1),
                                    dim_ordering='th',
                                    W_regularizer=l2(0.001),
                                    b_regularizer=l2(0.001),
                                    activity_regularizer=None,
                                    W_constraint=None,
                                    b_constraint=None))

        logging.debug("Layer3:Conv3D shape={0}".format(self._mdl.output_shape))
        self._mdl.add(Activation(LeakyReLU(alpha=0.1)))

        # max pool 1
        self._mdl.add(MaxPooling3D(pool_size=(2, 2, 2),
                                   strides=None,
                                   border_mode='valid',
                                   dim_ordering='th'))

        logging.debug("Layer4:MaxPool3D shape={0}".format(self._mdl.output_shape))

        # dropout 2
        self._mdl.add(Dropout(p=0.3))

        # dense 1 (fully connected layer)
        self._mdl.add(Flatten())
        logging.debug("Layer5:Flatten shape={0}".format(self._mdl.output_shape))

        self._mdl.add(Dense(output_dim=128,
                            init='normal',  # TODO np.random.normal, K.random_normal
                            activation='linear',
                            weights=None,
                            W_regularizer=l2(0.001),
                            b_regularizer=l2(0.001),
                            activity_regularizer=None,
                            W_constraint=None,
                            b_constraint=None))

        logging.debug("Layer6:Dense shape={0}".format(self._mdl.output_shape))

        # dropout 3
        self._mdl.add(Dropout(p=0.4))

        # dense 2 (fully connected layer)
        self._mdl.add(Dense(output_dim=1,
                            init='normal',  # TODO np.random.normal, K.random_normal
                            activation='linear',
                            weights=None,
                            W_regularizer=l2(0.001),
                            b_regularizer=l2(0.001),
                            activity_regularizer=None,
                            W_constraint=None,
                            b_constraint=None))

        logging.debug("Layer8:Dense shape={0}".format(self._mdl.output_shape))

        # TODO softmax needed ?
        self._mdl.add(Activation("softmax"))

        # compile model
        self._mdl.compile(loss=self._objective, optimizer=self._optimizer, metrics=["accuracy"])
        logging.info("Model compiled!")

    def _objective(self, y_true, y_pred):
        # TODO might need to use np_utils.to_categorical(y, nb_classes=None)
        return K.categorical_crossentropy(y_pred, y_true)

    def fit(self, generator, samples_per_epoch,
            nb_epoch, valid_generator, nb_valid_samples):
        self._mdl.fit_generator(generator=generator,
                                samples_per_epoch=samples_per_epoch,
                                nb_epoch=nb_epoch,
                                verbose=1,
                                callbacks=[],
                                validation_data=valid_generator,
                                nb_val_samples=nb_valid_samples,
                                class_weight=None)

        # TODO more sophisticated filename
        self._mdl.save_weights("weights", False)

    # for testing only
    def _fit(self, X_train, y_train, batch_size=32, nb_epoch=80):
        self._mdl.fit(X_train=X_train,
                      y_train=y_train,
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      shuffle=True,
                      verbose=1)

        self._mdl.save_weights("weights", overwrite=False)

    def evaluate(self, evaluation_generator, num_eval_samples):
        self._score = self._mdl.evaluate_generator(
                                         generator=evaluation_generator,
                                         val_samples=num_eval_samples)
        print("Test score:", self._score)

    def load_weights(self, file):
        logging.info("Loading model weights from file '{0}'".format(file))
        self._mdl.load_weights(file)

    def predict(self, X_predict):
        # TODO add prediction from PointCloud
        self._mdl.predict(X_predict)

    def get_score(self):
        # TODO might change to tuble(self._score), not sure it's tuple
        return self._score

    score = property(get_score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    v = model_vt()
    loader = lib_IO_hdf5.Loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/testing.hdf5",
                                     set_type= "train",
                                     batch_size= 12,
                                     shuffle=True,
                                     valid_split=0.15,
                                     mode="train")
    v.fit(generator=loader.train_generator(),
          samples_per_epoch=loader.return_num_train_samples(),
          nb_epoch=2,
          valid_generator= loader.valid_generator(),
          nb_valid_samples = loader.return_num_valid_samples())
    # v.load_weights("weightsm")

    v.evaluate(evaluation_generator = loader.evaluate_generator(),
               num_eval_samples=loader.return_num_evaluation_samples())
