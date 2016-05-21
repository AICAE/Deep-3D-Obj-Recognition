#!/usr/bin/python2
# -*- coding: utf-8 -*-

import model_keras
import vtk
import lib_IO_hdf5
from config import model_cfg
import model_keras
import logging
import numpy as np
import sys
import time
import os
import argparse
from mayavi import mlab
import pcl
import random


def main():
    parser = argparse.ArgumentParser(description="Run voxnet object recognizer")

    parser.add_argument("file",
                        help="file, which holds object that should be recognized")

    parser.add_argument("-pc", "--is_point_cloud", action="store_true",
                        dest="is_point_cloud", help="boolean option for is point cloud")

    parser.add_argument("-w", "--weights", metavar="weights",
                        dest="weights_file", help="use given weights file")

    parser.add_argument("-c", "--num_classes", type=int, default=39,
                        dest="num_classes", help="use given weights file")

    # parse args
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        # TODO replace with logging
        logging.error("[!] File does not exist '{0}'".format(args.dataset))
        sys.exit(-1)


    #load Point cloud file
    pc = pcl.load(args.file)
    np_pc = np.asarray(pc)

class display():
    """

    use mayavi for point cloud

    """
    def __init__(self):
        #TODO
        mlab.show()

    def display_pc(self,np_pc):
        self.disp_pc = mlab.points3d(np_pc[:,0], np_pc[:,1], np_pc[:,2])

    def display_vox(self, np_vox):


        self.disp_vox = mlab.points3d(vox_x, vox_y, vox_z, mode='cube')

    def display_wait(self):
        #TODO

    def display_label(self):
        #TODO



class recognizer_voxnet:
    """

    use model_voxnet predict

    """
    def __init__(self, weights, nb_classes):
        self.mdl = model_keras.model_vt(nb_classes=nb_classes, dataset_name="zzz")
        self.mdl.load_weights(weights)

    def predict(self, X_pred, is_pc = False):
        if is_pc == True:
            X_pred = self.voxilize(X_pred)
        return self.mdl.predict(X_pred)

    def voxilize(self, np_pc, cam_pos = None):
        # chance to fill voxel
        p = 80

        max_dist = 0.0
        for it in range(0,2):
            # find min max & distance in current direction
            min = np.amin(np_pc[:,it], axis=0)
            max = np.amax(np_pc[:,it], axis=0)
            dist = max-min

            #find maximum distance
            if dist > max_dist:
                max_dist = dist

            #set middle to 0,0,0
            np_pc[:,it] = np_pc[:,it] - dist/2

        #find voxel edge size
        vox_sz = max_dist/30

        #render pc to size 30x30x30 from middle
        np_pc = np_pc()/vox_sz + max_dist/2

        #round to integer array
        np_pc = np.rint(np_pc).astype(np.uint32)

        #fill voxel arrays
        vox = np.zeros([30,30,30])
        for (pc_x, pc_y, pc_z) in np_pc:
            if random.randint(0,100) < 80:
                vox[pc_x, pc_y, pc_z] = 1

        if cam_pos is None:
            #TODO fill space between voxels

        else:
            #TODO estimate unknown space

        return vox


if __name__ == "__main__":
    main()