import h5py
import time
import logging
import sys

import lib_IO_hdf5
from config import model_cfg
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


##############test-saving####################################################


lib_IO_hdf5.save_dataset_as_hdf5("/home/tg/Downloads/volumetric_data/",
                           "/home/tg/Projects/Deep-3D-Obj-Recognition/src/data/testing.hdf5",
                          model_cfg.class_name_to_id_testing)

#################test_loading##########################################

#f = h5py.File("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5")


# it = 0
# for feat, lab in lib_IO_hdf5.Loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5",
#                                  set_type= "train",
#                                  batch_size= 12,
#                                  num_batches=20,
#                                  shuffle=True,
#                                  valid_split=0.15):
#     print("for the {0}. batch the shape is {1}".format(it,feat.shape))
#
#     it +=1


# it = 0
# for feat, lab in lib_IO_hdf5.Loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5",
#                                  set_type= "test",
#                                  batch_size= 12 * 64,
#                                  num_batches=2,
#                                  shuffle=False,
#                                  valid_split=0.15):
#     print("for the {0}. batch the shape is {1}".format(it,feat.shape))
#     it +=1

# tic = time.time()
#
# b = lib_IO_hdf5.Loader_hdf5_Convert_Np("data/modelnet40.hdf5",
#                                  batch_size= 32,
#                                  has_rot= False,
#                                  shuffle=True,
#                                  valid_split=0.15)
#
# print(b.return_nb_classes())
#
# print("-----------")
# it = 0
# for feat, label in b.train_generator():
#     print("it: {0} ---- feat: {1} ---- label: {2}".format(it, feat.shape,label.shape))
#     it += 1
#     if it >= 1000:
#         break
#
# print("-----------")
# it = 0
# for feat, label in b.valid_generator():
#     print("it: {0} ---- feat: {1} ---- label: {2}".format(it, feat.shape,label.shape))
#     it += 1
#     if it >= 1000:
#         break
#
# print("-----------")
# it = 0
# for feat, label in b.evaluate_generator():
#     print("it: {0} ---- feat: {1} ---- label: {2}".format(it, feat.shape,label.shape))
#     it += 1
#     if it >= 1000:
#         break
#
# print("-----------")
# for it in range(0,121,12):
#     print(b._info[it*12:(it+1)*12,1])
#
# tictoc = time.time() - tic
# print("the test_IO with Convert to Numpy took {0} seconds".format(tictoc))



# tic = time.time()
# with lib_IO_hdf5.Loader_hdf5("data/modelnet40.hdf5",
#                                  batch_size= 32*64,
#                                  has_rot= False,
#                                  shuffle=True,
#                                  valid_split=0.10) as b:
#
#     print("-----------")
#     it = 0
#     for feat, label in b.train_generator():
#         print("it: {0} ---- feat: {1} ---- label: {2}".format(it, feat.shape,label.shape))
#         it += 1
#         if it >= 1000:
#             break
#
#     print("-----------")
#     it = 0
#     for feat, label in b.valid_generator():
#         print("it: {0} ---- feat: {1} ---- label: {2}".format(it, feat.shape,label.shape))
#         it += 1
#         if it >= 1000:
#             break
#
#     print("-----------")
#     it = 0
#     for feat, label in b.evaluate_generator():
#         print("it: {0} ---- feat: {1} ---- label: {2}".format(it, feat.shape,label.shape))
#         it += 1
#         if it >= 1000:
#             break
#
#     print("-----------")
#     for it in range(0,121,12):
#         print(b._info[sorted(b._pos_train_indizes[it*12:12*(it+1)]),1])
#
#
# tictoc = time.time() - tic
# print("the test_IO without Conversion to Numpy run took {0} seconds".format(tictoc))