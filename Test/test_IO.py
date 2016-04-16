import h5py

from Source.voxnet_keras import lib_IO_hdf5
from Source.voxnet_keras.config import model_cfg

##############test-saving####################################################


lib_IO_hdf5.save_dataset_as_hdf5("/home/tg/Downloads/volumetric_data/",
                           "/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/testing.hdf5",
                           model_cfg.class_name_to_id_testing)

#################test_loading##########################################

f = h5py.File("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/testing.hdf5")
print(f["train/labels_train"].shape)
print(f["train/features_train"].shape)
print(f["test/labels_test"].shape)
print(f["test/features_test"].shape)


it = 0
for feat, lab in lib_IO_hdf5.loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/testing.hdf5",
                                 set_type= "train",
                                 batch_size= 12 * 128):
    print("for the {0}. batch the shape is {1}".format(it,feat.shape))
    it +=1



it = 0
for feat, lab in lib_IO_hdf5.loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/testing.hdf5",
                                 set_type= "test",
                                 batch_size= 12 * 64):

    print("for the {0}. batch the shape is {1}".format(it,feat.shape))
    it +=1


