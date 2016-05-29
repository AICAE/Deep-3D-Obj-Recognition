#!/usr/bin/python2
# -*- coding: utf-8 -*-


import model_keras
from config.model_cfg import class_id_to_name_modelnet40
import numpy as np
import random
from scipy.io import loadmat

def load_pc(fname):
    f = loadmat(fname)
    data = f['data'].astype(np.float32)
    return data

class detector_voxnet:
    """

    use model_voxnet predict

    """
    def __init__(self, weights, nb_classes = 39):
        self.mdl = model_keras.model_vt(nb_classes=nb_classes, dataset_name="modelnet")
        self.mdl.load_weights(weights)

    def predict(self, X_pred, is_pc = False):
        if is_pc == True:
            X_pred = self.voxilize(X_pred)
        label =  self.mdl.predict(X_pred)
        print("label {0}".format(label))
        #label = class_id_to_name_modelnet40(str(label))
        return label

    def voxilize(self, np_pc, rot = None):
        # chance to fill voxel
        p = 80

        max_dist = 0.0
        for it in range(0,3):
            # find min max & distance in current direction
            min = np.amin(np_pc[:,it])
            max = np.amax(np_pc[:,it])
            dist = max-min

            #find maximum distance
            if dist > max_dist:
                max_dist = dist

            #set middle to 0,0,0
            np_pc[:,it] = np_pc[:,it] - dist/2 - min

            #find voxel edge size
            vox_sz = dist/29

            #render pc to size 30x30x30 from middle
            np_pc[:,it] = np_pc[:,it]/vox_sz

        for it in range(0,3):
            np_pc[:,it] = np_pc[:,it] + 14.5

        #round to integer array
        np_pc = np.rint(np_pc).astype(np.uint32)

        #fill voxel arrays
        vox = np.zeros([30,30,30])
        for (pc_x, pc_y, pc_z) in np_pc:
            if random.randint(0,100) < 80:
                vox[pc_x, pc_y, pc_z] = 1

        if rot is None:
            a = 1
            #TODO fill space between voxels
        else:
            a = 1
            #TODO extra: create 12 rotations with unknown space and pool decision
            #TODO estimate unknown space

        #TODO create boundary
        np_vox = np.zeros([1,1,32,32,32])
        np_vox[0, 0, 1:-1, 1:-1, 1:-1] = vox

        return np_vox