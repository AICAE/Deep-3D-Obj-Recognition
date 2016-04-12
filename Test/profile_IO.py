import numpy as np
import time
import cProfile
from io import StringIO, BytesIO
import pstats


# #import mayavi.mlab as mlab
#
from Source.voxnet_keras import lib_IO
from Source.voxnet_keras.config import model_cfg

tic = time.time()
pr = cProfile.Profile()
pr.enable()
##############test-saving####################################################

f1,l1,f2,l2 = lib_IO.load_data("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.npz")

tictoc = time.time() - tic
print("elapsed time si {0}".format(tictoc))

pr.disable()
s = BytesIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())