#!/usr/bin/python3

import keras
assert keras.__version__ == "1.0.0", "keras version not supported"

from keras import backend as K

from keras.models import Sequential

from keras.layers import Convolution3D, MaxPooling3D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.engine.training import batch_shuffle

from keras.optimizers import SGD

import lib_IO_hdf5

import logging
import datetime



def learningRateSchedule(epoch):
    if epoch >= 60000:
        return 0.0001
    elif epoch >= 400000:
        return 0.00005,
    elif epoch >= 600000:
        return 0.00001
    else:
        return 0.001

class model_vt (object):
    def __init__(self,nb_classes):
        """initiate Model according to voxnet paper"""
        # Stochastic Gradient Decent (SGD) with momentum
        # lr=0.01 for LiDar dataset
        # lr=0.001 for other datasets
        # decay of 0.00016667 approx the same as learning schedule (0:0.001,60000:0.0001,600000:0.00001)
        # use callbacks learingrate_schedule instead
        self._optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        self._lr_schedule = LearningRateScheduler(learningRateSchedule)

        # init model
        self._mdl = Sequential()

        # convolution 1
        self._mdl.add(Convolution3D(input_shape=(1, 32, 32, 32),
                                    nb_filter=32,
                                    kernel_dim1=5,
                                    kernel_dim2=5,
                                    kernel_dim3=5,
                                    init='normal',
                                    border_mode='valid',
                                    subsample=(2, 2, 2),
                                    dim_ordering='th',
                                    W_regularizer=l2(0.001),
                                    b_regularizer=l2(0.001),
                                    ))

        logging.debug("Layer1:Conv3D shape={0}".format(self._mdl.output_shape))
        self._mdl.add(Activation(LeakyReLU(alpha=0.1)))

        # dropout 1
        self._mdl.add(Dropout(p=0.2))

        # convolution 2
        self._mdl.add(Convolution3D(nb_filter=32,
                                    kernel_dim1=3,
                                    kernel_dim2=3,
                                    kernel_dim3=3,
                                    init='normal',
                                    border_mode='valid',
                                    subsample=(1, 1, 1),
                                    dim_ordering='th',
                                    W_regularizer=l2(0.001),
                                    b_regularizer=l2(0.001),
                                    ))

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
                            init='normal',
                            activation='linear',
                            W_regularizer=l2(0.001),
                            b_regularizer=l2(0.001),
                            ))

        logging.debug("Layer6:Dense shape={0}".format(self._mdl.output_shape))

        # dropout 3
        self._mdl.add(Dropout(p=0.4))

        # dense 2 (fully connected layer)
        self._mdl.add(Dense(output_dim=nb_classes,
                            init='normal',
                            activation='linear',
                            W_regularizer=l2(0.001),
                            b_regularizer=l2(0.001),
                            ))

        logging.debug("Layer8:Dense shape={0}".format(self._mdl.output_shape))

        self._mdl.add(Activation("softmax"))

        # compile model
        self._mdl.compile(loss=self._objective, optimizer=self._optimizer, metrics=["accuracy"])
        logging.info("Model compiled!")

    def _objective(self, y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true)

    def fit(self, generator, samples_per_epoch,
            nb_epoch, valid_generator, nb_valid_samples):
        self._mdl.fit_generator(generator=generator,
                                samples_per_epoch=samples_per_epoch,
                                nb_epoch=nb_epoch,
                                verbose=1,
                                callbacks=[self._lr_schedule,],
                                validation_data=valid_generator,
                                nb_val_samples=nb_valid_samples,
                                )

        time_now = datetime.datetime.now()
        time_now = "_{0}_{1}_{2}_{3}_{4}_{5}".format(time_now.year, time_now.month, time_now.day,
                                                     time_now.hour, time_now.minute, time_now.second)
        logging.info("save model Voxnet weights as weights_{0}.h5".format(time_now))
        self._mdl.save_weights("weights_{0}.h5".format(time_now), False)

    # for testing only
    def _fit(self, X_train, y_train, batch_size=32, nb_epoch=80):
        self._mdl.fit(X_train=X_train,
                      y_train=y_train,
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      shuffle=True,
                      verbose=1,
                      )

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


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     v = model_vt()
#     loader = lib_IO_hdf5.Loader_hdf5("data/testing.hdf5",
#                                      batch_size= 12,
#                                      shuffle=True,
#                                      valid_split=0.15,
#                                      mode="train")
#     v.fit(generator=loader.train_generator(),
#           samples_per_epoch=loader.return_num_train_samples(),
#           nb_epoch=2,
#           valid_generator= loader.valid_generator(),
#           nb_valid_samples = loader.return_num_valid_samples())
#     # v.load_weights("weightsm")
#
#     v.evaluate(evaluation_generator = loader.evaluate_generator(),
#                num_eval_samples=loader.return_num_evaluation_samples())
