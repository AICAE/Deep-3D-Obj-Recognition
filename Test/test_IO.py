<<<<<<< HEAD
import h5py

from Source.voxnet_keras import lib_IO_hdf5
from Source.voxnet_keras.config import model_cfg

##############test-saving####################################################


lib_IO_hdf5.save_dataset_as_hdf5("/home/tg/Downloads/volumetric_data/",
                           "/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5",
                           model_cfg.class_name_to_id_modelnet10)

#################test_loading##########################################

f = h5py.File("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5")
print(f["train/labels_train"].shape)
print(f["train/features_train"].shape)
print(f["test/labels_test"].shape)
print(f["test/features_test"].shape)


it = 0
for feat, lab in lib_IO_hdf5.loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5",
                                 set_type= "train",
                                 batch_size= 12 * 128):
    print("for the {0}. batch the shape is {1}".format(it,feat.shape))
    it +=1



it = 0
for feat, lab in lib_IO_hdf5.loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5",
                                 set_type= "test",
                                 batch_size= 12 * 64):

    print("for the {0}. batch the shape is {1}".format(it,feat.shape))
    it +=1



=======
import numpy as np
import time
import cProfile
from io import StringIO
import pstats


# #import mayavi.mlab as mlab
#
from Source.voxnet_keras import lib_IO
from Source.voxnet_keras.config import model_cfg

tic = time.time()
pr = cProfile.Profile()
pr.enable()
##############test-saving####################################################
print(model_cfg.class_names_modelnet10)

lib_IO.save_dataset_as_npy("/home/tg/Downloads/volumetric_data/",
                           "/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet40.npz",
                           model_cfg.class_name_to_id_modelnet40)

tictoc = time.time() - tic
print("elapsed time si {0}".format(tictoc))

pr.disable()
s = StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())



##############test-loading###################################################
# fname_testset = '/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10_test.tar'
# fname_trainset = '/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10_train.tar'
#
# cfg = model_cfg.voxnet_cfg
#
# dataset, labels = lib_IO.load_and_format(fname_testset, cfg)
#
# print('dataset shape is {1}   -- should be .. x 1 x 32 x 32 x 32'.format(dataset.shape))
# print('dataset shape is {1}   -- should be .. x 2'.format(labels.shape))
#
# firstimg = dataset[np.random.randint(10,100),0,:,:,:]
# print(firstimg.shape)
#
# #fig = plt.figure()
>>>>>>> 2f35585a8a85a37050d32b70dc41702cb58b5fac

