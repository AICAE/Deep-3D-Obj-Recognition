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


def main():
    parser = argparse.ArgumentParser(description="Run voxnet object recognizer")

    parser.add_argument("file",
                        help="file, which holds object that should be recognized")

    parser.add_argument("-pc", "--is_point_cloud", action="store_true",
                        dest="is_point_cloud", help="boolean option for is point cloud")

    parser.add_argument("-w", "--weights", metavar="weights",
                        dest="weights_file", help="use given weights file")

    # parse args
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        # TODO replace with logging
        logging.error("[!] File does not exist '{0}'".format(args.dataset))
        sys.exit(-1)


class display():
    """

    use vtk for point cloud

    """
    def __init__(self):
        #TODO

    def display_pc(self):
        #TODO

    def display_vox(self):
        #TODO

    def display_wait(self):
        #TODO

    def display_label(self):
        #TODO


class recognizer_voxnet:
    """

    use model_voxnet predict

    """
    def __init__(self, weights):
        #TODO

    def recognize(self, file):
        #TODO

    def voxilize(self, pointcloud):
        #TODO





if __name__ == "__main__":
    main()