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

def voxilize(np_pc, rot = None):
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

        #covered cells
        cls = 29

        #find voxel edge size
        vox_sz = dist/(cls-1)

        #render pc to size 30x30x30 from middle
        np_pc[:,it] = np_pc[:,it]/vox_sz

    for it in range(0,3):
        np_pc[:,it] = np_pc[:,it] + (cls-1)/2

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

def voxel_scatter(np_vox):
    vox_scat = np.zeros([0,3], dtype= np.uint32)
    for x in range(0,np_vox.shape[2]):
        for y in range(0,np_vox.shape[3]):
            for z in range(0,np_vox.shape[4]):
                if np_vox[0,0,x,y,z] == 1.0:
                    arr_tmp = np.zeros([1,3],dtype=np.uint32)
                    arr_tmp[0,:] = (x,y,z)
                    vox_scat = np.concatenate((vox_scat,arr_tmp))
    return vox_scat

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
        proba_all =  self.mdl.predict(X_pred)
        #indices 0 is equal to class 2
        label = str(np.argmax(proba_all) + 2)
        label = class_id_to_name_modelnet40[label]
        proba = np.amax(proba_all)

        return label, proba