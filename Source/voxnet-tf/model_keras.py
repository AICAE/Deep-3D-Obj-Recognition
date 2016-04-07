#!/usr/bin/python3

from keras import backend as K

from keras.models import Sequential

from keras.layers import Convolution3D, MaxPooling3D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import SGD

import theano.tensor as T

import pdb

#TODO add for all parameters
#Find proper Optimizer
class model_vt (object):
    #initiate Model according to voxnet paper
    def __init__(self):
       
        # Stochastic Gradient Decent (SGD) with momentum
        # lr=0.01 for LiDar dataset
        # lr=0.001 for other datasets
        # TODO decay, learnrate reduction
        self._optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
        
        self._mdl= Sequential()
        #use shape
        #Convolution1

        self._mdl.add(Convolution3D(input_shape=(1, 32, 32, 32),
                                    nb_filter=32,
                                    kernel_dim1=5,
                                    kernel_dim2=5,
                                    kernel_dim3=5,
                                    init='normal', #TODO
                                    activation=LeakyReLU(alpha=0.1),
                                    weights=None, #TODO
                                    border_mode='valid',
                                    subsample=(2, 2, 2),
                                    dim_ordering='th', #TODO
                                    W_regularizer=None,
                                    b_regularizer=None,
                                    activity_regularizer=None,
                                    W_constraint=None,
                                    b_constraint=None))

        #Dropout1
        self._mdl.add(Dropout(p=0.2))
        
        #Convolution2
        self._mdl.add(Convolution3D(nb_filter=32,
                                    kernel_dim1=3,
                                    kernel_dim2=3,
                                    kernel_dim3=3,
                                    init='normal', #TODO
                                    activation=LeakyReLU(alpha=0.1),
                                    weights=None, #TODO
                                    border_mode='valid',
                                    subsample=(1, 1, 1),
                                    dim_ordering='th', #TODO
                                    W_regularizer=None,
                                    b_regularizer=None,
                                    activity_regularizer=None,
                                    W_constraint=None,
                                    b_constraint=None))

        #MaxPool1
        self._mdl.add(MaxPooling3D(pool_size=(2, 2, 2),
                                  strides=None,
                                  border_mode='valid',
                                  dim_ordering='th')) #TODO

        #Dropout2
        self._mdl.add(Dropout(p=0.3))
        
        
        #Dense1
        self._mdl.add(Dense(output_dim=128, # TODO not sure ...
                           init='glorot_uniform', #TODO zero_mean gauss
                           activation='linear',
                           weights=None,
                           W_regularizer=None,
                           b_regularizer=None,
                           activity_regularizer=None,
                           W_constraint=None,
                           b_constraint=None,
                           input_dim=None))
        
        #Dropout3
        self._mdl.add(Dropout(p=0.4))
        
        #Dense2
        self._mdl.add(Dense(output_dim=1, # TODO not sure ...
                   init='glorot_uniform', #TODO zero_mean gauss
                   activation='linear',
                   weights=None,
                   W_regularizer=None,
                   b_regularizer=None,
                   activity_regularizer=None,
                   W_constraint=None,
                   b_constraint=None,
                   input_dim=None))

        #Compile Model
        self._mdl.compile(loss=self._objective, optimizer=self._optimizer)


    def _objective(self, y_true, y_pred):
        #TODO replace!
        return K.mean(K.square(y_pred - y_true), axis=-1)


    def fit(self, X_train, y_train, cv=10):
        #TODO add cross-validation
        self.mdl.fit()


    def add_data(self):
        #TODO load old weigths
        #TODO improve fit with old weigths
        self.mdl.fit()


    def evaluate(self, X_test, y_test):
        #TODO make sure to use score from modelnet40/10 paper
        self.mdl.evaluate(X_test, y_test)


    def predict(self,X_predict):
        #TODO add prediction from PointCloud
        self.mdl.predict(X_predict)


model_vt()
