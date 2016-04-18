import h5py

import lib_IO_hdf5_test
from config import model_cfg

##############test-saving####################################################
#
#
# lib_IO_hdf5.save_dataset_as_hdf5("/home/tg/Downloads/volumetric_data/",
#                            "/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5",
#                           model_cfg.class_name_to_id_modelnet10)

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
#
b = lib_IO_hdf5_test.Loader_hdf5("data/testing.hdf5",
                                 batch_size= 12,
                                 num_batches=20,
                                 shuffle=True,
                                 valid_split=0.15)

print("-----------")
it = 0
for feat, label in b.train_generator():
    print(feat.shape)
    print(label.shape)
    it += 1
    if it >= 3:
        break

print("-----------")
it = 0
for feat, label in b.valid_generator():
    print(feat.shape)
    print(label.shape)
    it += 1
    if it >= 3:
        break

print("-----------")
it = 0
for feat, label in b.evaluate_generator():
    print(feat.shape)
    print(label.shape)
    it += 1
    if it >= 3:
        break

print("-----------")
for it in range(0,121,12):
    print(b._info[sorted(b._pos_train_indizes[it*12:12*(it+1)]),1])


