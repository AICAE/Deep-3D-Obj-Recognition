import sys
import os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__) + '/config/'))
import voxnet_cfg
import modelnet_cfg

def print_dir():
    print(os.path.dirname(os.path.realpath(__file__) + '/config/'))